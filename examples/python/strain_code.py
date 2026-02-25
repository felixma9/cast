
import numpy as np
from scipy.io import loadmat
import math
import matplotlib.pyplot as plt
import matplotlib.image as mping
import matplotlib.pyplot as plt

# change RFdata to np.int16 if not enough memory

#########################################################
####### ADD COMMENTS !!!!!!!!!!!!!!!!!!! ################
#########################################################

# Image Info
nBLOCK_NUM = 128
nBLOCK_LENGTH = 64
nLINE_NUM = 256
#nLINE_LENGTH = 8192 # 128*64
nSTART_BLOCK = 0
nEND_BLOCK = 128
nSTART_LINE = 0
nEND_LINE = 256
nBLOCK_SPACING = 4 # space between block centers
bEXT_SEARCH = 1
bLATERAL = 1
bMEDIAN = 1
nKERNEL_SIZE = 3

bSTRAIN_LSE = 1
SUBSAMPLING_2D = 1
# number of neighbor which is used in Least Square Strain Estimator, higher gives smoother but less resolution
NEIGHBOURS = 6
# strain attenuation compensation
m_nCOMPEN_BASE = -1
m_fCOMPEN_EXP  = 0
SIGN_PRESERVE = 0

SOPVar = 1
BLOCK_MATCHING = 1


#MAX_CORRELATION??

#---------------------------------------#
#       0    Line num      256 lines
#        -----------------
#       |                 | 
#       |                 | Line length = Block num * Block length
#       |                 | 
#       |                 |
#       |                 | 128 blocks
#        -----------------
#---------------------------------------#

# Least square axial strain estimation for the strain
def StrainLSE(dis, str, nOfB, spaceBB, amp, startBlk, endBlk):
	# calculate the strain by fitting a line with the Least Square Error on the displacement and 
	# report the lines slope as the strain on that region
    
    n = NEIGHBOURS	# the most important parameter which is the number of neighbors
    shift = 0		# move the averaging area little by little during the strain estimation to estimate the whole area

    k = startBlk

    for k in range(startBlk, endBlk):
        a=0
        c=0
        e=0
        f=0
        # the averaging window moves to left
        shift= (int)((float)((k-startBlk)*n)/ (float) (endBlk-startBlk))

        j = -shift
        # prepare variables using neighbouring values for this block
        for j in range (-shift, n-shift):
            a += j
            c += j*j
            e += dis[k+j]
            f += j*dis[k+j]
            j+=1
            
        # calculate strain using the Least Squares Slope Formula
        str[k] = amp * (a*e-n*f)/(a*a-n*c) / spaceBB;	
        
        # strain attenuation compensation; amplify deeper values
        if (m_nCOMPEN_BASE > 0) and (m_fCOMPEN_EXP > 0):
            str[k] = str[k] * pow(k/m_nCOMPEN_BASE+1.0, m_fCOMPEN_EXP) 
            
        # retuns the positive strain 0-> 0% and 2 ->2%
        if ((SIGN_PRESERVE==0) and str[k]<0):
            str[k] = -str[k];	
        
        k+=1

    return 0

def SumOfProduct(data1, data2, start1, start2, block_len, minExt, maxExt, lineLength, 
                 SOPVar=0, BLOCK_MATCHING=1):
    """Calculate correlation between RF windows with optional block matching.
    
    Args:
        SOPVar: 0=unnormalized, 1=normalized (NCC)
        BLOCK_MATCHING: 1=use 3 lines, 0=single line
    """
    # Boundary check
    if ((start1 < minExt) or (start2 < minExt) or 
        (start1 + block_len >= maxExt) or (start2 + block_len >= maxExt)):
        return 0.0
    
    # Current line
    preWin = data1[start1:start1+block_len]                        
    postWin = data2[start2:start2+block_len]      
    sum = np.dot(preWin, postWin)  
    autoCorrelation1 = np.dot(preWin, preWin)
    autoCorrelation2 = np.dot(postWin, postWin)

    if BLOCK_MATCHING == 1:
        # Previous line
        p_preWin = data1[start1-lineLength:start1-lineLength+block_len]  
        p_postWin = data2[start2-lineLength:start2-lineLength+block_len]  
        sump = np.dot(p_preWin, p_postWin)
        autoCorrelation1p = np.dot(p_preWin, p_preWin)
        autoCorrelation2p = np.dot(p_postWin, p_postWin)

        # Next line
        n_preWin = data1[start1+lineLength:start1+lineLength+block_len] 
        n_postWin = data2[start2+lineLength:start2+lineLength+block_len]
        sumn = np.dot(n_preWin, n_postWin)
        autoCorrelation1n = np.dot(n_preWin, n_preWin)
        autoCorrelation2n = np.dot(n_postWin, n_postWin)
        
        # Combine all three lines
        total_sum = sum + sump + sumn 
        total_corr1 = autoCorrelation1 + autoCorrelation1p + autoCorrelation1n
        total_corr2 = autoCorrelation2 + autoCorrelation2p + autoCorrelation2n
    else:
        # Single line only
        total_sum = sum
        total_corr1 = autoCorrelation1
        total_corr2 = autoCorrelation2
    
    # Return normalized or unnormalized
    if SOPVar == 0:    
        return total_sum
    else:
        denom = np.sqrt(total_corr1 * total_corr2)
        # Prevent division by zero
        if denom > 1e-10:
            total_sum / denom
        else:
            return 0.0

def SubPixelling(a, b, c):
    """
    Estimate sub-pixel peak location using cosine interpolation.
    
    Given three correlation values at positions [-1, 0, +1], estimates the 
    offset of the true peak from position 0.
    
    Args:
        a: Correlation at position -1
        b: Correlation at position 0 (should be the peak)
        c: Correlation at position +1
    
    Returns:
        Fractional offset in range [-1, 1]
        - Negative = peak shifted left (toward a)
        - Positive = peak shifted right (toward c)
        - Zero = peak exactly at b
    """
    # Validate that b is the maximum
    if b < a or b < c or b == 0:
        return 0.0
    
    # Calculate omega (curvature parameter)
    omega = math.acos((a + c) / (2 * b))
    
    # Check for invalid omega (NaN, infinity, or zero)
    if not np.isfinite(omega) or omega == 0:
        return 0.0
    
    # Calculate theta (asymmetry parameter)
    theta = math.atan2(a - c, 2 * b * math.sin(omega))
    
    # Calculate fractional offset
    lam = -theta / omega
    
    # Clamp to valid range [-1, 1]
    lam = np.clip(lam, -1.0, 1.0)
    
    return lam

def CorrelationRecalculation(a, b, c):
    """
    Recalculate correlation at sub-pixel location.
    
    Args:
        a, b, c: Correlation at positions -1, 0, +1
        lam: Fractional offset from SubPixelling
    
    Returns:
        Interpolated correlation value at position lam
    """
    # check that b (the peak) is not 0
    if b == 0:
        return 0
    
    # Calculate omega (curvature parameter)
    omega = math.acos((a + c) / (2 * b))

    # Check for invalid omega (NaN, infinity, or zero)
    if not np.isfinite(omega) or omega == 0:
        return 0.0

    # Calculate theta (asymmetry parameter)
    theta = math.atan2(a - c, 2 * b * math.sin(omega))
    
    # check that theta is not 0
    if math.cos(theta) == 0:
        return 0
    
    # calculate corr
    corr = b/math.cos(theta)
    
    # Clamp to valid range [-1, 1]
    lam = np.clip(corr, -1.0, 1.0)
    
    return corr

def getCorrelationEstimator():
    return SOPVar

def setCorrelationEstimator(correlationEstimator):
    match correlationEstimator:
        case 0:
            SOPVar = 0
            minCorrelation = 1000
            maxCorrelation = 2000
        case 1:
            SOPVar = 1
            minCorrelation = 0.5
            maxCorrelation = 0.8
        case 2:
            SOPVar = 2
            minCorrelation = 1000000
            maxCorrelation = 2000000
        case 3:
            SOPVar = 3
            minCorrelation = 0.5
            maxCorrelation = 0.7
        case _:
            SOPVar = 0
            minCorrelation = 20000
            maxCorrelation = 40000

def medianFilter2():
    return 0

def TDPE(pPreBuffer,      # previous frame (2D array)
         pBuffer):         # current frame

    """
            axialDisp,       # axial displacement
            lateralDisp,     # lateral displacement or filtered axial displacement
            estimatedStrain, # strain image
            MS,              # correlation image
            searchMap):      # shows the search map
    """

	# IMAGEINFO lpImgInfo ???????????

    ## MAX_STRAIN ??

    RF1 = pPreBuffer
    RF2 = pBuffer
    lineLength = RF1.shape[0] #?????

    fAveNCC = 0.0
	
    k = 0 # line number
    i = 0 # block number
    j = 0 # lag number
	
    #unsigned int bStrainLSE = lpImgInfo->uStrainLSE;
	#float fAmplify = lpImgInfo->fAmplify;
    lineLeftMargin = max(1, nSTART_LINE)
    lineRightMargin = min(nLINE_NUM, nEND_LINE-5)
    blockLeftMargin = max(5, nSTART_BLOCK)	# Number of Windows at each RF line	
    blockRightMargin = min(nBLOCK_NUM, nEND_BLOCK-5)
    
    maximumSearch = 360		# MAXSTRAIN * nLineLength / 200 ?
    fullSearchEstimator = 1	# 0: non-normalized , 1: normalized correlation

    # margins for search region
    searchLeftMargin = 0			
    searchRightMargin = 0
    
    blockSpacing = nBLOCK_SPACING	# Image resolution or spacing between block centers
    initialOffset = int(blockSpacing/2) # 2
    offset = initialOffset + lineLeftMargin * lineLength	# required offset to go to the first of each line (start at 2, +4096, get to start of next line (4098))
    
    preDisplacement = 0		# lag from last block
    pos = 0   # current lag
    
    # variables for subpixelling quadratic parabola fitting (or cosine-based model?) and correlation recalculation
    a = 0.0  # value to the left of peak
    b = 0.0  # current peak
    c = 0.0  # value to the right of peak
    L = 0.0		# Lambda
    similarity = np.zeros(3 * maximumSearch, dtype=np.float32)	# buffer to save the previously calculated correlation to avoid recalculation (correlation of all lags for current block - from -360 to 360 - has extra space)
    
    increaseSearchSize = 2 # why?

    size = nLINE_NUM * nBLOCK_NUM

    searchMap = np.zeros(size, dtype=np.float32)
    lateralDisp = np.zeros(size, dtype=np.float32)
    axialDisp = np.zeros(size, dtype=np.float32)
    MS = np.zeros(size, dtype=np.float32)

    # search window radius around prediction
    threshold = 1

    k = lineLeftMargin
    i = blockLeftMargin

    for k in range(lineLeftMargin,lineRightMargin):
        preDisplacement = 0;  # initialize for each line
				
		# calculating the displacement with Novel Time Domain
        for i in range(blockLeftMargin, blockRightMargin): 			
			# a = 0; b = 0; c = 0; L = 0;
				
			# don't let the prediction move out of the search region
            if (preDisplacement > maximumSearch) or (preDisplacement < -maximumSearch): 
                if preDisplacement > 0: 
                    preDisplacement = maximumSearch
                if preDisplacement < -maximumSearch:
                    preDisplacement = -maximumSearch
			
			# prediction of search region
            searchLeftMargin  = preDisplacement - threshold
            searchRightMargin = preDisplacement + threshold

            j = searchLeftMargin

            # cross correlation in the search region for this block (i)
            for j in range(searchLeftMargin, searchRightMargin):
			    # search area in each block and save correlation in similarity array 
                similarity[j-searchLeftMargin] = SumOfProduct(RF1,RF2,(offset+i*blockSpacing),(offset+i*blockSpacing+j), nBLOCK_LENGTH, offset, offset+ lineLength - initialOffset, lineLength)
                # if current correlation > correlation stored for this block, overwrite with current correlation and store current lag in pos
                if (similarity[j-searchLeftMargin] > MS[k*nBLOCK_NUM + i]):
                    MS[k*nBLOCK_NUM + i] = similarity[j-searchLeftMargin]
                    pos = j
                j = j+1

            # if correlation > 0 ????? or minCorrelation? found for this block at lags within the search region (if we found a candidate)
            if (MS[k*nBLOCK_NUM + i]> 0): 

				# retrive the correlation coefficients of the neighboring windows for subpixelling
                # if pos is after left margin correlation has been calculated
                if (pos > searchLeftMargin):
                    a = similarity[(pos-1)-searchLeftMargin]
                # if pos = searchLeftMargin, need to calculate searchLeftMargin-1 to find left neighbour
                else:		
                    a = SumOfProduct(RF1,RF2,(offset + i * blockSpacing),(offset + i * blockSpacing+(pos-1)),nBLOCK_LENGTH,offset,offset + lineLength - initialOffset, lineLength)
                    
                # correlation at current lag
                b = similarity[(pos)-searchLeftMargin]						
                
                # if pos is before right margin, correlation has been calculated
                if (pos < searchRightMargin):
                    c = similarity[(pos+1)-searchLeftMargin]
                # if pos = searchRightMargin, need to calculate searchRightMargin+1 to find right neighbour
                else:
                    c = SumOfProduct(RF1,RF2,(offset + i * blockSpacing),(offset + i * blockSpacing+(pos+1)),nBLOCK_LENGTH,offset, offset + lineLength - initialOffset, lineLength)
                
                # if b is the peak (greater than neighbours)
                if (b>=a) and (b>=c):
					# apply subpixelling to increase the accuracy of motion estimation (can be a float between a,b,c)
                    L = SubPixelling(a, b, c)
					# add subpixeling information(lag) to axial displacement array 
                    axialDisp[k*nBLOCK_NUM + i] = (pos+L)
					# recalculate the correlation at peak location
                    MS[k*nBLOCK_NUM + i] = CorrelationRecalculation(a, b, c, L)
					# save the prediction for the next step
                    preDisplacement = round(pos + L)
                
                elif (increaseSearchSize>0): 
                    # if a is the peak, continue the search in the left direction
                    if (a>=b and a>=c):
                        cnt = 0
                        # continue until b is true peak
                        while ((b<a)or(b<c)and(cnt<maximumSearch/4)):
                            # move pos (lag) to the left (becomes lag at a)
                            pos = pos-1
                            # shift variables to the left
                            c = b
                            b = a
                            # find correlation of value to the left of a (pos-1)
                            a = SumOfProduct(RF1,RF2,(offset + i * blockSpacing),(offset + i * blockSpacing+(pos-1)),nBLOCK_LENGTH,offset, offset + lineLength - initialOffset, lineLength)
                            cnt = cnt+1
						# subpixelling to increase the accuracy of motion estimation
                        L = SubPixelling(a, b, c)
                        # subpixeling information is added
                        axialDisp[k*nBLOCK_NUM + i] = (pos+L);		
                        # recalculate the correlation at peak location
                        MS[k*nBLOCK_NUM + i] = CorrelationRecalculation(a, b, c, L)
                        # save the prediction for the next step
                        preDisplacement = round(pos + L);		
                        
                    # if c is the peak, continue the search in the right direction    
                    else:
                        cnt = 0
                        # continue until b is true peak (highest correlation)
                        while ((b<a)or(b<c)and(cnt<maximumSearch/4)):
                            # increase pos (pos = lag at c)
                            pos = pos+1
                            # shift variables to the right
                            a = b 
                            b = c
                            # find correlation of value to the right of c (pos+1)
                            c = SumOfProduct(RF1,RF2,(offset + i * blockSpacing),(offset + i * blockSpacing+(pos+1)),nBLOCK_LENGTH,offset, offset + lineLength - initialOffset, lineLength);
                            cnt = cnt+1
						# subpixelling to increase the accuracy of motion estimation
                        L = SubPixelling(a, b, c)
						# subpixeling information is added
                        axialDisp[k*nBLOCK_NUM + i] = (pos+L)
                        # recalculate the correlation at peak location
                        MS[k*nBLOCK_NUM + i] = CorrelationRecalculation(a, b, c, L)
                        # save the prediction for the next step
                        preDisplacement = round(pos + L)
            
            # if correlation not high enough for this block at all lags within the search region (if we did not find a candidate) and are allowed to run a full search
            # i+k % 2 -- to increase computational efficiency????
            # maxCorrelation?
            if ((bEXT_SEARCH)and((i+k) % 2 == 0)and(MS[k*nBLOCK_NUM + i] < maxCorrelation)):
				# Begin ********* to increase the search region *************
				
				# mark this window in the full search map as an indicator of recovery search ?
                searchMap[k*nBLOCK_NUM + i] = 1
                # full search estimator can be different than original estimator to save some time
                currentEstimator = getCorrelationEstimator()
                setCorrelationEstimator(fullSearchEstimator); 
                
                # set correlation of this block to 0 
                MS[k*nBLOCK_NUM + i] = 0

                # search region - deeper tissue has larger displacement (region increases with block number i)
                searchLGLeftMargin = -3 - maximumSearch*i/nBLOCK_NUM
                searchLGRightMargin = 3 + maximumSearch*i/nBLOCK_NUM
                
                j = searchLGLeftMargin
                for j in range(searchLGLeftMargin, searchLGRightMargin): 
                    # search area in each block and save in some array to be used for subpixelling
                    similarity[j-searchLGLeftMargin] = SumOfProduct(RF1,RF2,(offset+i*blockSpacing),(offset+i*blockSpacing+j),nBLOCK_LENGTH,offset, offset + lineLength - initialOffset, lineLength);
                    if (similarity[j-searchLGLeftMargin] > MS[k*nBLOCK_NUM + i]):
                        MS[k*nBLOCK_NUM + i] = similarity[j-searchLGLeftMargin]
                        pos = j

                # set estimator back to what it was before full search
                setCorrelationEstimator(currentEstimator); 
				# reset the subpixelling parameters - necessary?
                a = 0
                b = 0
                c = 0
                L = 0
                
                if (currentEstimator == fullSearchEstimator):
                # we have already calculated the correlation values
                    if (pos>searchLGLeftMargin):
                        a = similarity[(pos-1)-searchLGLeftMargin]
                    b = similarity[(pos)-searchLGLeftMargin]
                    if (pos<searchLGRightMargin):
                        c = similarity[(pos+1)-searchLGLeftMargin]
                # if not, recalculate correlation with initial estimator
                else:
                    a = SumOfProduct(RF1,RF2,(offset + i * blockSpacing),(offset + i * blockSpacing+(pos-1)),nBLOCK_LENGTH,offset, offset + lineLength - initialOffset, lineLength)
                    b = SumOfProduct(RF1,RF2,(offset + i * blockSpacing),(offset + i * blockSpacing+(pos)),nBLOCK_LENGTH,offset, offset + lineLength - initialOffset, lineLength)
                    c = SumOfProduct(RF1,RF2,(offset + i * blockSpacing),(offset + i * blockSpacing+(pos+1)),nBLOCK_LENGTH,offset, offset + lineLength - initialOffset, lineLength)
                    
                # subpixelling to increase the accuracy of motion estimation
                L = SubPixelling(a, b, c)
                # subpixeling information is added
                axialDisp[k*nBLOCK_NUM + i] = (pos+L)
                # recalculate the correlation at peak location
                MS[k*nBLOCK_NUM + i] = CorrelationRecalculation(a, b, c, L)
                # save the prediction for the next step
                preDisplacement = round(pos + L);		
        
                # End *********** to increase the search region *************
            
            # Lateral motion tracking
            if((bLATERAL)and(k != 0)and(k != nLINE_NUM)):
                position = preDisplacement
                # neighboring RF lines
                offsetLeft  = offset - lineLength
                offsetRight = offset + lineLength

                # do 1D SUBSAMPLING
                if (SUBSAMPLING_2D == 0):
                    #         Left line (k-1)    Current line (k)    Right line (k+1)
                    #               ↓                                        ↓
                    # position-1:  [a1]                                    [a2]
                    # position:    [b1]          [Already found]           [b2]
                    # position+1:  [c1]                                    [c2]

                    # test correlation at LEFT neighbor line with 3 axial offsets
                    # LEFT line, axial lag = position-1
                    a1 = SumOfProduct(RF1,RF2,(offset + i * blockSpacing),(offsetLeft + i * blockSpacing+(position-1)),nBLOCK_LENGTH,offsetLeft, offset + lineLength - initialOffset, lineLength)
                    # LEFT line, axial lag = position (best axial match)
                    b1 = SumOfProduct(RF1,RF2,(offset + i * blockSpacing),(offsetLeft + i * blockSpacing+(position)),nBLOCK_LENGTH,offsetLeft, offset + lineLength - initialOffset, lineLength)
                    # LEFT line, axial lag = position+1
                    c1 = SumOfProduct(RF1,RF2,(offset + i * blockSpacing),(offsetLeft + i * blockSpacing+(position+1)),nBLOCK_LENGTH,offsetLeft, offset + lineLength - initialOffset, lineLength)
                    
                    # test correlation at RIGHT neighbor line with 3 axial offsets  
                    # RIGHT line, axial lag = position-1
                    a2 = SumOfProduct(RF1,RF2,(offset + i * blockSpacing),(offsetRight + i * blockSpacing+(position-1)),nBLOCK_LENGTH,offsetLeft, offsetRight + lineLength - initialOffset, lineLength)
                    # RIGHT line, axial lag = position (best axial match)
                    b2 = SumOfProduct(RF1,RF2,(offset + i * blockSpacing),(offsetRight + i * blockSpacing+(position)),nBLOCK_LENGTH,offsetLeft, offsetRight + lineLength - initialOffset, lineLength)
                    # RIGHT line, axial lag = position+1
                    c2 = SumOfProduct(RF1,RF2,(offset + i * blockSpacing),(offsetRight + i * blockSpacing+(position+1)),nBLOCK_LENGTH,offsetLeft, offsetRight + lineLength - initialOffset, lineLength)
                    
                    # find true peak amplitude at left line
                    a = CorrelationRecalculation(a1, b1, c1, L)
                    # current correlation
                    b = MS[k*nBLOCK_NUM + i]
                    # find true peak amplitude at right line
                    c = CorrelationRecalculation(a2, b2, c2, L)
                    
                    # if b is the true peak 
                    if ((b>=a)and(b>=c)):
                        # get exact value (can be between left/center or center/right)
                        L = SubPixelling(a, b, c)
                        lateralDisp[k*nBLOCK_NUM + i] = 0.5 + L
                        # ?????
                        MS[k*nBLOCK_NUM + i] = a*(L)*(L-1)/2 - b*(1+L)*(L-1) + c*(1+L)*(L)/2
                    
                    # shift to the left is the true peak (lateral displacement is 0)
                    elif ((a>=b)and(a>=c)):
                        lateralDisp[k*nBLOCK_NUM + i] = 0
                        MS[k*nBLOCK_NUM + i] = a
                    # shift to the right is the true peak (lateral displacement is 1)
                    else:
                        lateralDisp[k*nBLOCK_NUM + i] = 1
                        MS[k*nBLOCK_NUM + i] = c

                # do 2D SUBSAMPLING 
                A3 = a
                A1 = b
                A2 = c
                
                A5 = SumOfProduct(RF1,RF2,(offset + i * blockSpacing),(offsetLeft  + i * blockSpacing+(position)),nBLOCK_LENGTH,offsetLeft, offset      + lineLength - initialOffset, lineLength)
                A4 = SumOfProduct(RF1,RF2,(offset + i * blockSpacing),(offsetRight + i * blockSpacing+(position)),nBLOCK_LENGTH,offsetLeft, offsetRight + lineLength - initialOffset, lineLength)
                A6 = SumOfProduct(RF1,RF2,(offset + i * blockSpacing),(offsetRight + i * blockSpacing+(position+1)),nBLOCK_LENGTH,offsetLeft, offsetRight + lineLength - initialOffset, lineLength)
                
                if ((b>=a)and(b>=c)):
                    #subPixelling2D(&La, &Ll, A1, A2, A3, A4, A5, A6)
                    lateralDisp[k*nBLOCK_NUM + i] = 0.5f + Ll
                
                elif ((a>=b)and(a>=c)):
                    lateralDisp[k*nBLOCK_NUM + i] = 0
                    MS[k*nBLOCK_NUM + i] = a
                    
                else:
                    lateralDisp[k*nBLOCK_NUM + i] = 1
                    MS[k*nBLOCK_NUM + i] = c
		   
           	# Sum of NCC
            fAveNCC += MS[k*nBLOCK_NUM + i]

        # End of each block/window
        i = i+1
        
        # correct the offsets for the next step
        offset += lineLength

    # End of each line
    k = k+1

    # 2D median filter to remove possible errors in speckle tracking - which values does it do this for??
    if(bMEDIAN):
        winSize = nKERNEL_SIZE
        medianFilter2(axialDisp, nBLOCK_NUM, nLINE_NUM, searchMap, winSize, lineLeftMargin, lineRightMargin, blockLeftMargin, blockRightMargin)
        axialDisp[:] = searchMap

        # if we have both axial and lateral displacements
        if(bLATERAL):	
            medianFilter2(lateralDisp, nBLOCK_NUM, nLINE_NUM, searchMap, winSize, lineLeftMargin, lineRightMargin, blockLeftMargin, blockRightMargin)
            lateralDisp[:] = searchMap

	# Strain Estimation
    # derivative of axial displacement in axial direction 
    if(bSTRAIN_LSE == 1): 		
        # k = line counter
        k = lineLeftMargin
        for k in range(lineLeftMargin, lineRightMargin):
            StrainLSE(axialDisp + k*nBLOCK_NUM, estimatedStrain+k*nBLOCK_NUM, nBLOCK_NUM, blockSpacing, fAmplify,blockLeftMargin,blockRightMargin)
            k = k+1
            
    fAveNCC = fAveNCC/(lineRightMargin-lineLeftMargin)/(blockRightMargin-blockLeftMargin)
    return fAveNCC

