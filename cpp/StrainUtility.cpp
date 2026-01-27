// StrainUtility.cpp: implementation of the StrainUtility class.
//
//////////////////////////////////////////////////////////////////////

#include "StrainUtility.h"
#include <math.h>
#include <float.h>

#define PIX_SORT(a,b) { if ((a)>(b)) PIX_SWAP((a),(b)); }
#define PIX_SWAP(a,b) { unsigned char temp=(a);(a)=(b);(b)=temp; }

//#define   SUBSAMPLING2D		1

#define	BLOCKMATCHING		1 

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

StrainUtility::StrainUtility()
{
	SOPVar = 1;			//0: sum of product, 1:nomrmalized sop, 2:sop with log compression 3:sop with sign
	neighbours = 6;		// number of neighbor which is used in Least Square Strain Estimator, higher gives smoother but less resolution

	counter = 1;
	threshold = 1;
	averaging = 80;

	setCorrelationEstimator(SOPVar);

	for( int cn = 0; cn < NUMOFBUFFER; cn++ )
	{
		m_buffer[cn] = NULL;
		bufferCounter[cn] = -1;
	}

	m_nLineNum = 0;
	m_nBlockNum = 0;

	m_fCompenExp = 0;			// For strain attenuation compensation
	m_nCompenBase = -1;

	signPreserve = false;
}

StrainUtility::~StrainUtility()
{
	DeleteAll();
}

float StrainUtility::SumOfProduct(short* data1,short* data2,int start1,int start2,int len,int minExt, int maxExt, int lineLength)
{	// calculates the sum of product between two vectors
	// from start point of each vector up to start+length
	if ((start1<minExt) || (start2<minExt) || (start1+len>=maxExt)|| (start2+len>=maxExt))
		return float(0.0f);

	int i;
	double sum =0, autoCorrelation1=0, autoCorrelation2=0;
	short  *PreRFwin  = &data1[start1];
	short  *PostRFwin = &data2[start2];

#ifdef	BLOCKMATCHING

	double sump=0, autoCorrelation1p=0, autoCorrelation2p=0;
	short  *PreRFwinp  = &data1[start1 - lineLength];	// previous A line
	short  *PostRFwinp = &data2[start2 - lineLength];
	
	double sumn=0, autoCorrelation1n=0, autoCorrelation2n=0;
	short  *PreRFwinn  = &data1[start1 + lineLength];	// next A line
	short  *PostRFwinn = &data2[start2 + lineLength];
#endif
	

	if (SOPVar == 0)
	{
		for (i=0; i<len; i++)
		{
#ifdef	BLOCKMATCHING
			sum += (*PreRFwin++) * (*PostRFwin++);
			sump += (*PreRFwinp++) * (*PostRFwinp++);	// previous A line
			sumn += (*PreRFwinn++) * (*PostRFwinn++);	// next A line
#endif
#ifndef	BLOCKMATCHING			
			sum += data1[start1++]*data2[start2++];
#endif

		}
#ifdef	BLOCKMATCHING
		return (float) (sum + sumn + sump);
#endif
		return (float) sum;
	}else
	{
		for (i=0; i<len; i++)
		{
			short v1 = *PreRFwin++;
			short v2 = *PostRFwin++;
			sum += v1*v2;
			autoCorrelation1 += v1*v1;
			autoCorrelation2 += v2*v2;

#ifdef	BLOCKMATCHING


			short v1p = *PreRFwinp++;	// previous A-line
			short v2p = *PostRFwinp++;
			sump += v1p*v2p;
			autoCorrelation1p += v1p*v1p;
			autoCorrelation2p += v2p*v2p;

			short v1n = *PreRFwinn++;	// next A line
			short v2n = *PostRFwinn++;
			sumn += v1n*v2n;
			autoCorrelation1n += v1n*v1n;
			autoCorrelation2n += v2n*v2n;
#endif
		}
#ifdef	BLOCKMATCHING
		return float((sum + sump + sumn)/sqrt( (autoCorrelation1 + autoCorrelation1p + autoCorrelation1n)*(autoCorrelation2 + autoCorrelation2p + autoCorrelation2n)));
#endif
		return float(sum/sqrt( autoCorrelation1 * autoCorrelation2 ));
	}
}

// Least square axial strain estimation for the strain
void StrainUtility::StrainLSE(float *dis,float *str, int nOfB, int spaceBB, float amp, int startBlk, int endBlk)
{	// caculate the strain by fitting a line with the Least Square Error on the displacement and 
	// report the lines slope as the strain on that region

	int n=neighbours;	//the most important parameter which is the number of neighbors
	int shift=0;		// move the averaging area little by little during the strain estimation to estimate the whole area
	float a=0,c=0,e=0,f=0;

	for (int k=startBlk; k<endBlk; k++)			// for first half blocks
	{
		a=0;
		c=0;
		e=0;
		f=0;
		shift= (int)((float)((k-startBlk)*n)/ (float) (endBlk-startBlk));		// the averaging window moves to left
		for (int j=-shift; j<n-shift; j++)
		{
			a += (j);
			c += (j)*(j);
			e += dis[k+j];
			f += (j)*dis[k+j];
		}

		str[k] = amp * (a*e-n*f)/(a*a-n*c) / spaceBB;	

		if( m_nCompenBase > 0 && m_fCompenExp > 0 )
			str[k] = (float)( str[k] * pow((float)(k)/(float)m_nCompenBase+1.0f, m_fCompenExp) );

		if (!signPreserve && str[k]<0)
			str[k] = -str[k];	//retuns the positive strain 0-> 0% and 2 ->2%
	}
	return;
}

void StrainUtility::medianFilter2(float *inp, int xlength, int ylength, float *outp, int winSize, int startLine, int endLine, int startBlock ,int endBlock)
{
	float* window = new float [winSize*winSize];
	int i,j,k,le,be;
	int middle = (int)(winSize/2); 
	float index;

	for (le=startLine; le<endLine; le++)
		for (be=startBlock; be<endBlock; be++)
			// for the boundary repeat the data
			if ((be < middle) || ((xlength - be)< middle) || (le < middle) || ((ylength - le)< middle) )
				outp[le*xlength + be] = inp[le*xlength + be];
			else {
				// for the rest of pixels replace with the median of the window
				for (k=le - middle; k<(le - middle + winSize); k++)
					memcpy(&window[(k - (le - middle))*winSize],&inp[k*xlength + be - middle],winSize*sizeof(float));

				// sort the window
				for (i=1; i < winSize*winSize; i++)
				{
					index = window[i];
					j = i;
					while ((j > 0) && (window[j-1] > index))
					{
					  window[j] = window[j-1];
					  j = j - 1;
					}
					window[j] = index;
				}

				// median
				outp[le*xlength + be] = window[middle * winSize + middle];
			}
	delete [] window;

}

void StrainUtility::setCorrelationEstimator(int correlationEstimator){
	// The user can decide which estimator he wants to used to do motion tracking
	// 0: non normamlized correlation 1: normalized correlation and 2: nomalized covariance

	switch(correlationEstimator)
	{
	case 0:		//non normalized correlation
		SOPVar = 0;
		minCorrelation = 1000;
		maxCorrelation = 2000;
		break;

	case 1:		// normalized correlation
		SOPVar = 1;
		minCorrelation = 0.5f;
		maxCorrelation = 0.8f;
		break;
	case 2:		// sum of squared differences
		SOPVar = 2;
		minCorrelation = 1000000;
		maxCorrelation = 2000000;
		break;
	case 3:
		SOPVar = 3;		// normalized covariance
		minCorrelation = 0.5f;
		maxCorrelation = 0.7f;
		break;
	
	default:
		SOPVar = 0;
		minCorrelation = 20000;
		maxCorrelation = 40000;
		break;
	}
}

int StrainUtility::getCorrelationEstimator(){
	return SOPVar;
}

float StrainUtility::maxCorrFilter2(float *inp, float *corr, int xlength, int ylength, float *outp, int startLine, int endLine, int startBlock , int endBlock , int mode)
{	// Thresholds the output by looking at its corresponding correlation map

	int le,be;
	//int numberOfError =0;
	float averageCorr = 0.0f;
	
	switch(mode)
	{
	case 0:	// no filtering
		memcpy(outp,inp,ylength*xlength*sizeof(float));	
		return 0;
	case 1:
		for (le=startLine; le<endLine ; le++)
			for (be=startBlock ; be<endBlock; be++)
			{
				if (!(corr[le*xlength + be]> maxCorrelation))
					outp[le*xlength + be] = 0;
				else
					outp[le*xlength + be] = inp[le*xlength + be] * corr[le*xlength + be];
				
				averageCorr = (corr[le*xlength + be] >=0.0f) ? averageCorr + corr[le*xlength + be] : averageCorr ;
			}
		return (float)averageCorr/(float)((endBlock-startBlock)*(endLine-startLine));	
	}
	return 0;
}

void StrainUtility::clearTemporalBuffer()
{
	for(int cn = 0; cn < NUMOFBUFFER; cn++)
	{
		if (m_buffer[cn])
			delete [] m_buffer[cn];
		m_buffer[cn] = NULL;
		bufferCounter[cn] = -1;
	}

	m_nLineNum = 0;
	m_nBlockNum = 0;

	return;
}

// Reset the buffer used for temporal filtering
void StrainUtility::Reset(int nLineNum, int nBlockNum, BOOL bKalman)
{
	// ASSERT( nLineNum > 0 && nBlockNum > 0 );

	if( (m_nLineNum == nLineNum) && (m_nBlockNum == nBlockNum) )
		return;

	// Clear all the buffers for temporal filtering
	DeleteAll();

	m_nLineNum = nLineNum;
	m_nBlockNum = nBlockNum;
	counter = 0;

	// Use the temporal LSE
	for(int cn = 0; cn < NUMOFBUFFER; cn++)
		m_buffer[cn] = new unsigned char[BUFFERSIZE*nLineNum*nBlockNum];
}

float StrainUtility::subPixelling(float a, float b, float c)
{
	// coise fit
	if( b<a || b<c || b==0 )	// b should be the maximum
		return 0.0f;

	double omega = acos((a+c)/(2*b));
	if( _isnan(omega) || omega == 0 )	// NaN or divided by 0
		return 0;
	double theta = atan2(a-c, 2*b*(float)sin(omega));
	float lambda = float(-theta/omega);
	if( lambda > 1 )
		lambda = 1.0;
	if( lambda < -1 )
		lambda = -1.0;
	return lambda;
}

int StrainUtility::round(float data)
{
	return (data>0) ? (int)(data + 0.4999f) : (int)(data - 0.4999f);
}

void StrainUtility::DeleteAll()
{
	clearTemporalBuffer();
}

float StrainUtility::CorrelationRecalculation(float a, float b, float c, float L)
{
	//return a*(L)*(L-1)/2 - b*(1+L)*(L-1) + c*(1+L)*(L)/2;

	if( b == 0 )		// Divided by 0
		return 0;
	double omega = acos((a+c)/(2*b));
	if( _isnan(omega) )		// NaN
		return 0;
	double theta = atan2((float)(a-c), (float)(2*b*sin(omega)));
	if( cos(theta) == 0 )	// Divided by 0
		return 0;
	float corr = float(b/cos(theta));
	if( corr > 1 )
		corr = 1.0;
	else if( corr < -1 )
		corr = -1.0;

	return corr;

}

//--------------------------------------------------------------------------------
// Time domain cross-correlation with prior estimates (TDPE)
// derived from TDCPEFinal()
// all the parameters of the image info are packed in a structure
// return the percent of valid tissue motion estimates in the region of interest
//--------------------------------------------------------------------------------
float StrainUtility::TDPE(BYTE *pPreBuffer,			// previous frame
						  BYTE *pBuffer,			// current frame
						  float *axialDisp,			// Axial displacement
						  float *lateralDisp,		// Lateral displacement or filtered axial displacement
						  float *estimatedStrain,	// strain image
						  float *MS,				// correlation image
						  float *searchMap,			// shows the search map
						  LPRFIMAGEINFO lpImgInfo)
{
	// Time domain cross correlation with prior estimates
	signed short * RF1 = (signed short*)pPreBuffer;
	signed short * RF2 = (signed short*)pBuffer;

	float fAveNCC = 0.0f;

	int i, j, k;
	int numberOfBlocks = lpImgInfo->nBlockNum;
	int blockLength = lpImgInfo->nBlockLength;
	int nLineCount = lpImgInfo->nLineNum;
	unsigned int bStrainLSE = lpImgInfo->uStrainLSE;
	float fAmplify = lpImgInfo->fAmplify;

//	int lineLeftMargin = max(5, lpImgInfo->nStartLine);				// Number of RF Lines
	int lineLeftMargin = max(1, lpImgInfo->nStartLine);
	int lineRightMargin = min(nLineCount, lpImgInfo->nEndLine-5);
//	int lineRightMargin = min(nLineCount, lpImgInfo->nEndLine-1); //SARAM - April 2011
	int blockLeftMargin = max(5, lpImgInfo->nStartBlock);	// Number of Windows at each RF line	
//	int blockLeftMargin = max(0, lpImgInfo->nStartBlock);
	int blockRightMargin = min(numberOfBlocks, lpImgInfo->nEndBlock-5); //SARAM - April 2011
//	int blockRightMargin = min(numberOfBlocks, lpImgInfo->nEndBlock);
	int lineLength = lpImgInfo->nLineLength/2;

	int const maximumSearch = 360;		// MAXSTRAIN * nLineLength / 200;4 //RV: changed from 36 to 360
	int const fullSearchEstimator = 1;	// 0: non-normalized , 1: normalized correlation

	int leftMargin = 0;			// for search region
	int rightMargin = 0;
	
	int blockSpacing = lpImgInfo->nBlockSpacing;	// Image resolution or spacing between block centers
	int initialOffset = int(blockSpacing/2);
	int offset = initialOffset + lineLeftMargin * lineLength;	// required offset to go to the first of each line
	//RV:added this line to try w/o second term of above
	//int offset = initialOffset;
	
	int preDisplacement=0;		
	int	pos = 0;

	float a,b,c;	//  variable used for subpixelling with quadratic parabola fitting and correlation recalculation
	float L;		// Lambda
	float similarity[3*maximumSearch];		// buffer to save the previously calculated correlation to avoid recalculation
	
	int increaseSearchSize = 2;

	memset(searchMap,   0, sizeof(float) * nLineCount * numberOfBlocks);
	memset(lateralDisp,  0, sizeof(float) * nLineCount * numberOfBlocks);
	memset(axialDisp,  0, sizeof(float) * nLineCount * numberOfBlocks);
	memset(MS,            0, sizeof(float) * nLineCount * numberOfBlocks);

	for(  k = lineLeftMargin; k < lineRightMargin; k++ )		//k: line counter
	{	
		preDisplacement = 0;		//initialize
				
		//calculating the displacement with Novel Time Domain
		for( i=blockLeftMargin; i<blockRightMargin; i++ )		// i: window counter
		{	
			a = 0; b = 0; c = 0; L = 0;
				
			// don't let the prediction move out of the search region
			if ((preDisplacement>  maximumSearch) || (preDisplacement<  -maximumSearch) )
				preDisplacement = (preDisplacement > 0) ? maximumSearch : -maximumSearch;
			
			// prediction of search region
			leftMargin  = preDisplacement - threshold;
			rightMargin = preDisplacement + threshold;

			// cross correlation over the search region
			for (j=leftMargin; j<=rightMargin; j++) 
			{	//search area in each block and save in some array (similarity)
				similarity[j-leftMargin] = SumOfProduct(RF1,RF2,(offset+i*blockSpacing),(offset+i*blockSpacing+j), blockLength, offset, offset+ lineLength - initialOffset, lineLength);
				if (similarity[j-leftMargin]> MS[k*numberOfBlocks + i]) 
				{
					MS[k*numberOfBlocks + i] = (float)similarity[j-leftMargin];
					pos = (j);
				}
			}	

			if (MS[k*numberOfBlocks + i]> 0)	// if we found any candidate
			{
				// retrive the correlation coefficients of the neighboring windows for subpixelling
				if (pos>leftMargin)	// it has been calculated previously
					a = similarity[(pos-1)-leftMargin];
				else				// it has not been calculated previously => should be calculated again
					a = SumOfProduct(RF1,RF2,(offset + i * blockSpacing),(offset + i * blockSpacing+(pos-1)),blockLength,offset, offset + lineLength - initialOffset, lineLength);

				b = similarity[(pos)-leftMargin];	// it has been calculated for sure							

				if (pos<rightMargin)	// it has been calculated previously
					c = similarity[(pos+1)-leftMargin];
				else					// it has not been calculated previously
					c = SumOfProduct(RF1,RF2,(offset + i * blockSpacing),(offset + i * blockSpacing+(pos+1)),blockLength,offset, offset + lineLength - initialOffset, lineLength);

				if( (b>=a) && (b>=c) )
				{
					// subpixelling to increase the accuracy of motion estimation
					L = subPixelling(a, b, c);
					// subpixeling information is added
					axialDisp[k*numberOfBlocks + i] = (pos+L);		
					// recalculate the correlation at peak location
					MS[k*numberOfBlocks + i] = CorrelationRecalculation(a, b, c, L);
					// save the prediction for the next step
					preDisplacement = round(pos + L);		
				
				}
				else if (increaseSearchSize>0)
				{
					if (a>=b && a>=c)	// continue the search in left direction 
					{
						//leftcounter++;
						int cnt = 0;
						while ( (b<a) || (b<c) && (cnt<maximumSearch/4))
						{
							pos--;
							c = b; 
							b = a;
							a = SumOfProduct(RF1,RF2,(offset + i * blockSpacing),(offset + i * blockSpacing+(pos-1)),blockLength,offset, offset + lineLength - initialOffset, lineLength);							cnt++;
						}
						// subpixelling to increase the accuracy of motion estimation
						L = subPixelling(a, b, c);
						// subpixeling information is added
						axialDisp[k*numberOfBlocks + i] = (pos+L);		
						// recalculate the correlation at peak location
						MS[k*numberOfBlocks + i] = CorrelationRecalculation(a, b, c, L);
						// save the prediction for the next step
						preDisplacement = round(pos + L);		
					
					}
					else	// continue the search in right direction 
					{
						//leftcounter++;
						int cnt = 0;
						while ( (b<a) || (b<c) && (cnt<maximumSearch/4))
						{
							pos++;
							a = b; 
							b = c;
							c = SumOfProduct(RF1,RF2,(offset + i * blockSpacing),(offset + i * blockSpacing+(pos+1)),blockLength,offset, offset + lineLength - initialOffset, lineLength);
							cnt++;
						}
						// subpixelling to increase the accuracy of motion estimation
						L = subPixelling(a, b, c);
						// subpixeling information is added
						axialDisp[k*numberOfBlocks + i] = (pos+L);		
						// recalculate the correlation at peak location
						MS[k*numberOfBlocks + i] = CorrelationRecalculation(a, b, c, L);
						// save the prediction for the next step
						preDisplacement = round(pos + L);		
					}
				}
			}
			
			if ( (lpImgInfo->bExtSearch) && ( (i+k) % 2 == 0)&& MS[k*numberOfBlocks + i] < maxCorrelation)	
			{	// we didn't find any candidate and we are allowed to run full saerch
				// Begin ********* to increase the search region *************
				
				// mark this window in the full search map as an indicator of recovery search
				searchMap[k*numberOfBlocks + i] = 1;
				// full search estimator can be different than original estimator to save some time
				int currentEstimator = getCorrelationEstimator();

				setCorrelationEstimator(fullSearchEstimator); 
				MS[k*numberOfBlocks + i] = 0;

				// search region
				leftMargin = -3 - maximumSearch*i/numberOfBlocks;
				rightMargin = 3 + maximumSearch*i/numberOfBlocks;
				
				for (j=leftMargin; j<=rightMargin; j++) 
				{	//search area in each block and save in some array (diff) to be used for subpixelling
					similarity[j-leftMargin] = SumOfProduct(RF1,RF2,(offset+i*blockSpacing),(offset+i*blockSpacing+j),blockLength,offset, offset + lineLength - initialOffset, lineLength);
					if (similarity[j-leftMargin] > MS[k*numberOfBlocks + i]) 
					{
						MS[k*numberOfBlocks + i] = (float)similarity[j-leftMargin];
						pos = (j);
					}
				}	

				// set it back to what it was
				setCorrelationEstimator(currentEstimator); 
				// reset the subpixelling parameters
				a = 0; b = 0; c = 0; L = 0;
				
				if (currentEstimator == fullSearchEstimator)
				{	// we have already calculated the correlation values
					if (pos>leftMargin)
						a = similarity[(pos-1)-leftMargin];
					b = similarity[(pos)-leftMargin];
					if (pos<rightMargin)
						c = similarity[(pos+1)-leftMargin];
				}
				else
				{	// we need to recaculate the correlation values with initial estimator
					a = SumOfProduct(RF1,RF2,(offset + i * blockSpacing),(offset + i * blockSpacing+(pos-1)),blockLength,offset, offset + lineLength - initialOffset, lineLength);
					b = SumOfProduct(RF1,RF2,(offset + i * blockSpacing),(offset + i * blockSpacing+(pos)),blockLength,  offset, offset + lineLength - initialOffset, lineLength);
					c = SumOfProduct(RF1,RF2,(offset + i * blockSpacing),(offset + i * blockSpacing+(pos+1)),blockLength,offset, offset + lineLength - initialOffset, lineLength);
				}
				
				// subpixelling to increase the accuracy of motion estimation
				L = subPixelling(a, b, c);
				// subpixeling information is added
				axialDisp[k*numberOfBlocks + i] = (pos+L);		
				// recalculate the correlation at peak location
				MS[k*numberOfBlocks + i] = CorrelationRecalculation(a, b, c, L);
				// save the prediction for the next step
				preDisplacement = round(pos + L);		

				// End *********** to increase the search region *************
			}

			// Lateral motion tracking
			if( lpImgInfo->bLateral && (k != 0) && (k != nLineCount))
			{
				int position = preDisplacement;
				// neighboring RF lines
				int offsetLeft  = offset - lineLength;
				//int offsetLeft2 = offset - 2*lineLength;
				int offsetRight = offset + lineLength;
				//int offsetRight2= offset + 2*lineLength;

#ifndef SUBSAMPLING2D
				float a1,b1,c1,a2,b2,c2;

				a1 = SumOfProduct(RF1,RF2,(offset + i * blockSpacing),(offsetLeft + i * blockSpacing+(position-1)),blockLength, offsetLeft, offset + lineLength - initialOffset, lineLength);
				b1 = SumOfProduct(RF1,RF2,(offset + i * blockSpacing),(offsetLeft + i * blockSpacing+(position))  ,blockLength, offsetLeft, offset + lineLength - initialOffset, lineLength);
				c1 = SumOfProduct(RF1,RF2,(offset + i * blockSpacing),(offsetLeft + i * blockSpacing+(position+1)),blockLength, offsetLeft, offset + lineLength - initialOffset, lineLength);

				a2 = SumOfProduct(RF1,RF2,(offset + i * blockSpacing),(offsetRight + i * blockSpacing+(position-1)),blockLength, offsetLeft, offsetRight + lineLength - initialOffset, lineLength);
				b2 = SumOfProduct(RF1,RF2,(offset + i * blockSpacing),(offsetRight + i * blockSpacing+(position))  ,blockLength, offsetLeft, offsetRight + lineLength - initialOffset, lineLength);
				c2 = SumOfProduct(RF1,RF2,(offset + i * blockSpacing),(offsetRight + i * blockSpacing+(position+1)),blockLength, offsetLeft, offsetRight + lineLength - initialOffset, lineLength);

				a = CorrelationRecalculation(a1, b1, c1, L);
				b = MS[k*numberOfBlocks + i];
				c = CorrelationRecalculation(a2, b2, c2, L);

				if ((b>=a) && (b>=c))
				{
					L = subPixelling(a, b, c);
					lateralDisp[k*numberOfBlocks + i] = 0.5f + L;
					MS[k*numberOfBlocks + i] = a*(L)*(L-1)/2 - b*(1+L)*(L-1) + c*(1+L)*(L)/2;//CorrelationRecalculation(a, b, c, L);

				}else if ((a>=b) && (a>=c))
				{
					lateralDisp[k*numberOfBlocks + i] = 0;
					MS[k*numberOfBlocks + i] = a;
				}else
				{
					lateralDisp[k*numberOfBlocks + i] = 1;
					MS[k*numberOfBlocks + i] = c;
				}
#endif

#ifdef SUBSAMPLING2D
				float A1,A2,A3,A4,A5,A6,La,Ll;

				A3 = a;
				A1 = b;
				A2 = c;

				A5 = SumOfProduct(RF1,RF2,(offset + i * blockSpacing),(offsetLeft  + i * blockSpacing+(position))  ,blockLength, offsetLeft, offset      + lineLength - initialOffset, lineLength);
				A4 = SumOfProduct(RF1,RF2,(offset + i * blockSpacing),(offsetRight + i * blockSpacing+(position))  ,blockLength, offsetLeft, offsetRight + lineLength - initialOffset, lineLength);
				A6 = SumOfProduct(RF1,RF2,(offset + i * blockSpacing),(offsetRight + i * blockSpacing+(position+1)),blockLength, offsetLeft, offsetRight + lineLength - initialOffset, lineLength);

				if ((b>=a) && (b>=c))
				{
					subPixelling2D(&La, &Ll, A1, A2, A3, A4, A5, A6);
					lateralDisp[k*numberOfBlocks + i] = 0.5f + Ll;
					//axialDisp[k*numberOfBlocks + i] = (pos+La);		

				}else if ((a>=b) && (a>=c))
				{
					lateralDisp[k*numberOfBlocks + i] = 0;
					MS[k*numberOfBlocks + i] = a;
				}else
				{
					lateralDisp[k*numberOfBlocks + i] = 1;
					MS[k*numberOfBlocks + i] = c;
				}
#endif
			}

			// Sum of NCC
			fAveNCC += MS[k*numberOfBlocks + i];

		}// end of each block or window 

		// correct the offsets for the next step
		offset += lineLength;

	}// end of each line

	// 2D median filter to remove possible errors in speckle tracking
	if( lpImgInfo->bMedian )
	{
		int winSize = lpImgInfo->nKernelSize;
		medianFilter2(axialDisp, numberOfBlocks, nLineCount, searchMap, winSize, lineLeftMargin, lineRightMargin, blockLeftMargin, blockRightMargin);
		memcpy(axialDisp, searchMap, sizeof(float) * nLineCount * numberOfBlocks);
		if( lpImgInfo->bLateral )	// we have both axial and lateral displacements
		{
			medianFilter2(lateralDisp, numberOfBlocks, nLineCount, searchMap, winSize, lineLeftMargin, lineRightMargin, blockLeftMargin, blockRightMargin);
			memcpy(lateralDisp, searchMap, sizeof(float) * nLineCount * numberOfBlocks);
		}
	}

	// Strain Estimation
	if( bStrainLSE == 1 )		// derivative of axial displacement in axial direction 
	{
		for(  k = lineLeftMargin; k < lineRightMargin; k++)		//k: line counter
			StrainLSE(axialDisp + k*numberOfBlocks, estimatedStrain+k*numberOfBlocks, numberOfBlocks, blockSpacing, fAmplify,blockLeftMargin,blockRightMargin);
	}

	fAveNCC = fAveNCC/(lineRightMargin-lineLeftMargin)/(blockRightMargin-blockLeftMargin);
	return fAveNCC;
}
