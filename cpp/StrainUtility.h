// StrainUtility.h: interface for the StrainUtility class.
//
//////////////////////////////////////////////////////////////////////

#include "RFImageInfo.h"

#define BUFFERSIZE			8 //RV: changed from 8 to 16
#define NUMOFBUFFER			2

class StrainUtility  
{
public:
	StrainUtility();
	virtual ~StrainUtility();

	void medianFilter2(float *inp, int xlength, int ylength, float *outp, int winSize,int startLine, int endLine, int startBlock ,int endBlock );
	void clearTemporalBuffer();

public:
	float TDPE(BYTE *pPreBuffer, BYTE *pBuffer, float *displacement, float *filteredDisplacement, float *estimatedStrain, float *MS, float *m_SearchMap, LPRFIMAGEINFO lpImgInfo);
	void DeleteAll();
	void Reset(int nLineNum, int nBlockNum, BOOL bKalman = false);
	void StrainLSE(float *dis,float *str, int nOfB, int spaceBB, float norm, int startBlk, int endBlk);

	float maxCorrFilter2(float *inp, float *corr, int xlength, int ylength, float *outp, int startLine, int endLine, int startBlock , int endBlock , int mode);

	void setCorrelationEstimator(int correlationEstimator);
	int getCorrelationEstimator();
	float SumOfProduct(short* data1,short* data2,int start1,int start2,int len,int minExt, int maxExt, int lineLength);	
	int round(float data);

	int threshold;
	
	int averaging;

	int neighbours;
	int SOPVar;
	
	float minCorrelation;
	float maxCorrelation;

	float m_fCompenExp;
	int   m_nCompenBase;

	bool signPreserve;
	
private:

	// Strain image size
	int m_nLineNum;				// replace nLines
	int m_nBlockNum;			// replace nLength

	int counter;

	unsigned char *m_buffer[NUMOFBUFFER];			// temporal buffer filter
	int bufferCounter[NUMOFBUFFER];

	float subPixelling(float a, float b, float c);
	float CorrelationRecalculation(float a, float b, float c, float L);
};