import numpy as np
import cvxpy as cp
import scipy as scipy
import cvxopt as cvxopt

class utils:
    def MaxMinNorm(self):
        self = (self - max(self))/(max(self) - min(self)) + 2
        return(self)
         
    def Recc_ED_sm(df, th_level, sm):
        ndim = len(df)
        lend = len(df[0])
        smll = []
        smrl = []

        x = np.zeros([lend, lend])
        for i in range(lend):
            sm_left = i - sm
            if sm_left < 0:
                sm_left = 0
            
            sm_right = i + sm
            
            if sm_right > lend:
                sm_right = lend
            smll.append(sm_left)
            smrl.append(sm_right)
            
            for j in range(lend):
                if sm_left <= j and j <= sm_right:
                    for k in range(ndim):
                        x[i,j] += np.square(df[k][i] - df[k][j])
        
        x = np.sqrt(x)
        #th = np.quantile(x.reshape(-1), 0.75)
        th = np.max(x)*th_level
        
        vfunc = np.vectorize( lambda x : 0 if (x < th) else 1 )
        x_01 = vfunc(x)
        return (x, x_01)
    
    #
    def Recc_ED_sm_1(df, sm):
        ndim = len(df)
        lend = len(df[0])
        smll = []
        smrl = []

        x = np.zeros([lend, lend])
        for i in range(lend):
            sm_left = i - sm
            if sm_left < 0:
                sm_left = 0
            
            sm_right = i + sm
            
            if sm_right > lend:
                sm_right = lend
            smll.append(sm_left)
            smrl.append(sm_right)
            
            for j in range(lend):
                if sm_left <= j and j <= sm_right:
                    for k in range(ndim):
                        x[i,j] += np.square(df[k][i] - df[k][j])
        
        x = np.sqrt(x)
        return (x)

    def Recc_BD_sm(df, sm):
        ndim = len(df)
        lend = len(df[0])
        smll = []
        smrl = []

        x = np.zeros([lend, lend])
        for i in range(lend):
            sm_left = i - sm
            if sm_left < 0:
                sm_left = 0
            
            sm_right = i + sm
            
            if sm_right > lend:
                sm_right = lend
            smll.append(sm_left)
            smrl.append(sm_right)
            
            for j in range(lend):
                top, bot1, bot2 = 0, 0, 0
                if sm_left <= j and j <= sm_right:
                    for k in range(ndim):
                        top += df[k-1][i]*df[k-1][j]
                        bot1 += df[k][i]
                        bot2 += df[k][j]
                    
                try:
                    x[i,j] += 1 - ( top/(bot1*bot2) )
                except ZeroDivisionError:
                    x[i,j] += 0
                
        
        #x = np.sqrt(x)
        #th = np.quantile(x.reshape(-1), 0.75)
        #th = np.max(x)*th_level
        #print(np.max(x), th)
        
        #vfunc = np.vectorize( lambda x : 0 if (x <= th) else 1 )
        #x_01 = vfunc(x)
        return (x)
   
    def Recc_GADF_sm(df):
        len_df = df.shape[1]
        #############################################
        ### USE RAW DF, NOT NORMALIZED OR PAA YET ###
        ### ALSO DELETE THE COLUMN IF NOT FEATURE ###
        #############################################
        from sklearn.decomposition import TruncatedSVD
        from pyts.image import GramianAngularField
        # define array
        df_temp = np.array(df.copy()).T
        # using svd to combine time series to 1
        svd = TruncatedSVD(n_components=1)
        svd.fit(df_temp)
        result = svd.transform(df_temp)
        # normalize array from svd
        df = np.array(result).T
        df_normd = np.apply_along_axis(utils.MaxMinNorm, 1, df)
        df_normd = df_normd[0]
        # Gramian Angular Field
        #gasf = GramianAngularField(image_size=dataset_size, method='summation')
        gadf = GramianAngularField(image_size=len_df, method='difference')
        #a_gasf = gasf.fit_transform(df_normd.reshape(1,-1))
        a_gadf = gadf.fit_transform(df_normd.reshape(1,-1))
        # make GADF range start from 0
        b_gadf = np.array(a_gadf[0]+abs(np.min(a_gadf[0])))
        # heaviside function
        #th = np.max(b_gadf)*th_level
        #vfunc = np.vectorize( lambda x : 0 if (x < th_level) else 1 )
        #c_gadf = vfunc(b_gadf)
        '''
        fghi = np.array(a_gasf[0]+abs(np.min(a_gasf[0])))
        th = np.max(fghi)*th
        vfunc = np.vectorize( lambda x : 0 if (x < th) else 1 )
        fghij = vfunc(fghi)
        '''
        return (b_gadf)
    
    def l1_trend_filtering(df):
        y = df
        n = y.size

        # Form second difference matrix.
        e = np.ones((1, n))
        D = scipy.sparse.spdiags(np.vstack((e, -2*e, e)), range(3), n-2, n)

        # Set regularization parameter.
        vlambda = 50

        # Solve l1 trend filtering problem.
        x = cp.Variable(shape=n)
        obj = cp.Minimize(0.5 * cp.sum_squares(y - x) + vlambda * cp.norm(D@x, 1) )
        prob = cp.Problem(obj)

        # ECOS and SCS solvers fail to converge before
        # the iteration limit. Use CVXOPT instead.
        prob.solve(solver=cp.CVXOPT, verbose=False)
        #print('Solver status: {}'.format(prob.status))

        # Check for error.
        #if prob.status != cp.OPTIMAL:
        #    raise Exception("Solver did not converge!")

        return (x.value)
    
    def calculateLREC(RP, w, th, min_length, gdd_version):
        LREC = np.array([0]*(w-1))
        for p in range((w-1),len(RP)):
            LREC = np.append(LREC, np.mean(RP[p-w+1:(p+1),p-w+1:(p+1)]))
        # LREC = the LREC vector
        
        vfunc = np.vectorize( lambda x : 0 if (x <= np.mean(LREC)) else 1 )
        LREC_bin = vfunc(LREC)
        LREC_bin = np.append(0, LREC_bin)
        # LREC_bin = vectorized LREC
        
        sub_id = np.array([])
        for i in range(1, len(LREC_bin)):
            if LREC_bin[i-1] != LREC_bin[i] :
                sub_id = np.append(sub_id, i)

        if np.max(sub_id) < len(LREC) :
            sub_id = np.append(sub_id, len(LREC))

        # eliminating the short sub_id
        low = np.array([])
        for i in range(1, len(sub_id)):
            if (sub_id[i] - sub_id[i-1]) < min_length : # short sub_id
                low = np.append(low, sub_id[i])
        sub_id = sub_id[~np.isin(sub_id, low)]
        sub_id = sub_id.astype(int)
        sub_id = np.append(0, sub_id)
        # sub_id = break point of LREC corresponding to the subsequences
        
        if gdd_version == 1 : # gdd version 1
            # Feature 1 and 2 : Mean and Length
            mean_lrec = np.array([])
            length_lrec = np.array([])
            area_lrec = np.array([])
            last = 0

            for i in range(1, len(sub_id)):
                f1 = np.mean( LREC[(sub_id[i-1]):sub_id[i]] )
                f2 = len( LREC[(sub_id[i-1]):sub_id[i]] )
                mean_lrec = np.append(mean_lrec, f1)
                length_lrec = np.append(length_lrec, f2)

            #1. Normalizing the subsequence data
            sub_features = np.vstack((mean_lrec, length_lrec))
            sub_features_normd = np.apply_along_axis(utils.MaxMinNorm, 1, sub_features)
            #sub_features_normd = sub_features

            #2. Create the recurrence plot
            # 75% quantile
            x11 = utils.Recc_ED_sm(df = sub_features_normd, th_level = th, sm=w+5)
            x12 = utils.Recc_GADF_sm(df = sub_features_normd, th_level = th)

            x13 = x11[0] + np.abs(x12[0]-1)
            #vfunc = np.vectorize( lambda x : 0 if (x < 2) else 1 )
            #x14 = vfunc(x13)
            x14 = x13/np.max(x13)

            #3. Calculate the gdd
            gdd = np.mean(x14, axis=1)
            gdd[0] = gdd[-1] = np.median(gdd)

        if gdd_version == 2 : # gdd version 2
            subseq = {}
            for i in range(1, len(sub_id)):
                a  = utils.l1_trend_filtering( df_normd[0][sub_id[i-1]:sub_id[i]] )
                b  = utils.l1_trend_filtering( df_normd[1][sub_id[i-1]:sub_id[i]] )
                subseq[i-1]  = np.append(a,b).reshape(2,a.size)
        
        
        return (LREC, LREC_bin, sub_id, gdd)
    
    
    
    