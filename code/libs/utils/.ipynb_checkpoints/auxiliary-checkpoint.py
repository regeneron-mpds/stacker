# auxiliary functions 
# cropAdata, padAdata, scaleData, iterModel, scale_spatial_coords2, makeContourImg, getCountourImage2

from PIL import Image
import numpy as np
import pandas as pd
import ants
import SimpleITK as sitk
from libs.utils.image.tissue import standard_mask_fn
from libs.alignment import sc_register, ants_align_pil2

from libs.utils.image import (pad_img as pad_img_to_even_tile,
                          unpad_img as unpad_img_from_even_tile,
                          stitch,
                          resize_to_max_dim_pil)
from libs.utils.transform import (scale_affine,
                              ants2itk_affine,
                              apply_itk_trf_image,
                              channeltransform,
                              register_spatial_with_itk_points,
                              register_spatial_with_itk_raster,
                              register_spatial_with_def_field_points,
                              register_spatial_with_def_field_raster)


def scale_spatial_coords2(X, ref_min=None, ref_max=None, max_val=10.0):
    if ref_min is None:
        ref_min = X.min(0)
        #print("ref_min = "+str(ref_min))
    X = X - ref_min
    if ref_max is None:
        ref_max = X.max(0)
        #print("ref_max = "+str(ref_max))
    X = X / ref_max
    return X * max_val


def cropAdata(adata,spatial_key,library_id, padratio=0.1, doMask=False,withOuts=False,dev4search=0.2,ninterval4search=50):
    adata_trim = adata.copy()
    newobsm=adata_trim.obsm.copy() # spot coords at full TIFF resolution
    newobs=adata_trim.obs.copy()
    newobsm[spatial_key]=newobsm[spatial_key][list(np.where(newobs['in_tissue'].to_numpy()==1)[0])] # new obsm['spatial']
    newobs=newobs.loc[newobs['in_tissue']==1] #new obs
    scalelow=adata_trim.uns[spatial_key][library_id]['scalefactors']['tissue_lowres_scalef']
    scalehigh=adata_trim.uns[spatial_key][library_id]['scalefactors']['tissue_hires_scalef']
    scale_imgIntensity=adata_trim.uns[spatial_key][library_id]['images']['hires']
    htif,wtif,ncolor=scale_imgIntensity.shape
    
    xy_indices=newobsm[spatial_key].copy()
    intissueMinx=(xy_indices[:,0].min())
    intissueMaxx=(xy_indices[:,0].max())
    intissueMiny=(xy_indices[:,1].min())
    intissueMaxy=(xy_indices[:,1].max())

    #@@ change to use tissue boundary not just in_tissue spot boundary to decide padding
    deltax=xy_indices[:,0].max()-xy_indices[:,0].min()
    deltay=xy_indices[:,1].max()-xy_indices[:,1].min()
    tissuelength=max(deltax,deltay)
    
    bestratioX, bestratioY=optPadlength3(tissuelength=tissuelength, padratio=padratio,spotCoordsTIF=xy_indices, 
                                         scale_imgIntensity=scale_imgIntensity,
                                        scalefactor=scalehigh, dev4search=dev4search, ninterval4search=ninterval4search)
    print((bestratioX, bestratioY))
    bestpadlengthX=min(min(round(tissuelength*bestratioX),xy_indices[:,0].min()), wtif/scalehigh-xy_indices[:,0].max())#padding length
    bestpadlengthY=min(min(round(tissuelength*bestratioY),xy_indices[:,1].min()), htif/scalehigh-xy_indices[:,1].max()) #padding length

#    padlength=min(round(tissuelength*padratio),min(xy_indices[:,0].min(),xy_indices[:,1].min())) #padding length

    newMinX=xy_indices[:,0].min()-bestpadlengthX # col?
    newMaxX=xy_indices[:,0].max()+bestpadlengthX
    newMinY=xy_indices[:,1].min()-bestpadlengthY # row?
    newMaxY=xy_indices[:,1].max()+bestpadlengthY

    
    #shift spot coords to match newly cropped img
    newobsm[spatial_key][:,0]=newobsm[spatial_key][:,0]-newMinX # = padlength
    newobsm[spatial_key][:,1]=newobsm[spatial_key][:,1]-newMinY


    hnewMinX=round(newMinX*scalehigh)+0 #PIL Image cropping is 0-index and end-exclusive
    hnewMaxX=round(newMaxX*scalehigh)+1
    hnewMinY=round(newMinY*scalehigh)+0
    hnewMaxY=round(newMaxY*scalehigh)+1


    lnewMinX=round(newMinX*scalelow)+0 #col/width
    lnewMaxX=round(newMaxX*scalelow)+1
    lnewMinY=round(newMinY*scalelow)+0 #row/height
    lnewMaxY=round(newMaxY*scalelow)+1


    #left, top, right, bottom
    if doMask:
        masked_lowres=maskNonTissue(adata_trim.uns[spatial_key][library_id]['images']['lowres'],intissueMinx,intissueMiny,intissueMaxx,intissueMaxy,None)
        masked_hires=maskNonTissue(adata_trim.uns[spatial_key][library_id]['images']['hires'],intissueMinx,intissueMiny,intissueMaxx,intissueMaxy,None)
        lcropped=Image.fromarray((masked_lowres * 255).astype(np.uint8)).convert('RGB').crop((lnewMinX,lnewMinY,lnewMaxX,lnewMaxY))
        hcropped=Image.fromarray((masked_hires* 255).astype(np.uint8)).convert('RGB').crop((hnewMinX,hnewMinY,hnewMaxX,hnewMaxY))
    else:
        lcropped=Image.fromarray((adata_trim.uns[spatial_key][library_id]['images']['lowres'] * 255).astype(np.uint8)).convert('RGB').crop((lnewMinX,lnewMinY,lnewMaxX,lnewMaxY))
        hcropped=Image.fromarray((adata_trim.uns[spatial_key][library_id]['images']['hires']* 255).astype(np.uint8)).convert('RGB').crop((hnewMinX,hnewMinY,hnewMaxX,hnewMaxY))

    adata_trim.uns[spatial_key][library_id]['images']['lowres']=(np.asarray(lcropped)/255).astype(np.float32)
    adata_trim.uns[spatial_key][library_id]['images']['hires']=(np.asarray(hcropped)/255).astype(np.float32)
    adata_trim.obs=newobs
    adata_trim.obsm=newobsm
    if withOuts:
        return [adata_trim, {'padlengthX':bestpadlengthX,'padlengthY':bestpadlengthY,'scalehigh':scalehigh,'tissuelength':tissuelength,
                             'hnewMinX':hnewMinX,'hnewMaxX':hnewMaxX,'hnewMinY':hnewMinY,'hnewMaxY':hnewMaxY,
                             'newMinX':newMinX,'newMaxX':newMaxX,'newMinY':newMinY,'newMaxY':newMaxY}]
    else:
        return adata_trim

def optPadlength(tissuelength, padratio, spotCoordsTIF,scale_imgIntensity, scalescalefactor, dev4search=0.2, ninterval4search=50):
    spotMinXtif=spotCoordsTIF[:,0].min()
    spotMinYtif=spotCoordsTIF[:,1].min()
    htif,wtif,ncolor=scale_imgIntensity.shape
    bestratioX=padratio
    #print((spotMinXtif,spotMinYtif))
    bestpadlengthX=min(min(round(tissuelength*bestratioX),spotMinXtif), wtif/scalefactor-spotCoordsTIF[:,0].max() ) #padding length
    bestdeltaX=abs((round(spotMinXtif*scalefactor)-round((spotMinXtif-bestpadlengthX)*scalefactor))-round(bestpadlengthX*scalefactor))

    bestratioY=padratio
    bestpadlengthY=min(min(round(tissuelength*bestratioY),spotMinYtif), htif/scalefactor-spotCoordsTIF[:,1].max()) #padding length
    bestdeltaY=abs((round(spotMinYtif*scalefactor)-round((spotMinYtif-bestpadlengthY)*scalefactor))-round(bestpadlengthY*scalefactor))
    for each in np.linspace(-padratio*dev4search, padratio*dev4search, ninterval4search):
        currentratioX=padratio+each
        currentpadlengthX=min(min(round(tissuelength*currentratioX),spotMinXtif), wtif/scalefactor-spotCoordsTIF[:,0].max())
        currentdeltaX=abs((round(spotMinXtif*scalefactor)-round((spotMinXtif-currentpadlengthX)*scalefactor))-round(currentpadlengthX*scalefactor))
        if currentdeltaX<bestdeltaX:
            bestdeltaX=currentdeltaX
            bestratioX=currentratioX                  
        currentratioY=padratio+each
        currentpadlengthY=min(min(round(tissuelength*currentratioY),spotMinYtif), htif/scalefactor-spotCoordsTIF[:,1].max())
        currentdeltaY=abs((round(spotMinYtif*scalefactor)-round((spotMinYtif-currentpadlengthY)*scalefactor))-round(currentpadlengthY*scalefactor))
        if currentdeltaY<bestdeltaY:
            bestdeltaY=currentdeltaY
            bestratioY=currentratioY                  
    return [bestratioX,bestratioY]    
    
    
def getSTimg(adata,spatial_key,library_id,img_key="hires"):
    return Image.fromarray((adata.uns[spatial_key][library_id]['images'][img_key]*255).astype(np.uint8))
    
def optPadlength2(tissuelength, padratio, spotCoordsTIF, scale_imgIntensity,scalefactor, dev4search=0.2, ninterval4search=50):
    spotMinXtif=spotCoordsTIF[:,0].min()
    spotMinYtif=spotCoordsTIF[:,1].min()
    htif,wtif,ncolor=scale_imgIntensity.shape

    bestratioX=padratio
    #print((spotMinXtif,spotMinYtif))
    bestpadlengthX=min(min(round(tissuelength*bestratioX),spotMinXtif), wtif/scalefactor-spotCoordsTIF[:,0].max() ) #padding length
    newMinX=spotCoordsTIF[:,0].min()-bestpadlengthX # col
    bestdeltaX=np.sum(np.abs((np.array(spotCoordsTIF[:,0])-newMinX)*scalefactor-np.round((np.array(spotCoordsTIF[:,0])-newMinX)*scalefactor)))

    bestratioY=padratio
    bestpadlengthY=min(min(round(tissuelength*bestratioY),spotMinYtif), htif/scalefactor-spotCoordsTIF[:,1].max()) #padding length
    newMinY=spotCoordsTIF[:,1].min()-bestpadlengthY # col
    bestdeltaY=np.sum(np.abs((np.array(spotCoordsTIF[:,1])-newMinY)*scalefactor-np.round((np.array(spotCoordsTIF[:,1])-newMinY)*scalefactor)))
    for each in np.linspace(-padratio*dev4search, padratio*dev4search, ninterval4search):
        currentratioX=padratio+each
        currentpadlengthX=min(min(round(tissuelength*currentratioX),spotMinXtif), wtif/scalefactor-spotCoordsTIF[:,0].max())
        newMinX=spotMinXtif-currentpaportdlengthX 
        currentdeltaX=np.sum(np.abs((np.array(spotCoordsTIF[:,0])-newMinX)*scalefactor-np.round((np.array(spotCoordsTIF[:,0])-newMinX)*scalefactor)))
        if currentdeltaX<bestdeltaX:
            bestdeltaX=currentdeltaX
            bestratioX=currentratioX         
            
        currentratioY=padratio+each
        currentpadlengthY=min(min(round(tissuelength*currentratioY),spotMinYtif), htif/scalefactor-spotCoordsTIF[:,1].max())
        newMinY=spotMinYtif-currentpadlengthY
        currentdeltaY=np.sum(np.abs((np.array(spotCoordsTIF[:,1])-newMinY)*scalefactor-np.round((np.array(spotCoordsTIF[:,1])-newMinY)*scalefactor)))
        if currentdeltaY<bestdeltaY:
            bestdeltaY=currentdeltaY
            bestratioY=currentratioY   
            
    return [bestratioX,bestratioY]    

def optPadlength3(tissuelength, padratio, spotCoordsTIF,scale_ImagegIntensity, scalefactor, dev4search=0.2, ninterval4search=50):
    refIntensity=np.array([scale_imgIntensity[round(spotCoordsTIF[x,1]*scalefactor),
                                     round(spotCoordsTIF[x,0]*scalefactor),:] 
                for x in range(spotCoordsTIF.shape[0])]).reshape(spotCoordsTIF.shape[0],3)
    htif,wtif,ncolor=scale_imgIntensity.shape

    spotMinXtif=spotCoordsTIF[:,0].min()
    spotMinYtif=spotCoordsTIF[:,1].min()
    bestspotCoords=spotCoordsTIF.copy()
    
    bestratioX=padratio
    bestpadlengthX=min(min(round(tissuelength*bestratioX),spotMinXtif), wtif/scalefactor-spotCoordsTIF[:,0].max() ) #padding length
    newMinX=bestspotCoords[:,0].min()-bestpadlengthX 
    newMaxX=bestspotCoords[:,0].max()+bestpadlengthX 
    hnewMinX=round(newMinX*scalefactor)+0 #PIL Image cropping is 0-index and end-exclusive
    hnewMaxX=round(newMaxX*scalefactor)+1
    bestspotCoords[:,0]=bestspotCoords[:,0]-newMinX
    
    bestratioY=padratio
    bestpadlengthY=min(min(round(tissuelength*bestratioY),spotMinYtif) , htif/scalefactor-spotCoordsTIF[:,1].max())#padding length
    newMinY=bestspotCoords[:,1].min()-bestpadlengthY # row
    newMaxY=bestspotCoords[:,1].max()+bestpadlengthY 
    hnewMinY=round(newMinY*scalefactor)+0
    hnewMaxY=round(newMaxY*scalefactor)+1
    bestspotCoords[:,1]=bestspotCoords[:,1]-newMinY
    
    hcropped=Image.fromarray((scale_imgIntensity.copy()* 255).astype(np.uint8)).convert('RGB').crop((hnewMinX,hnewMinY,hnewMaxX,hnewMaxY))
    hcropped=(np.asarray(hcropped)/255).astype(np.float32)
    bestIntensity=np.array([hcropped[round(bestspotCoords[x,1]*scalefactor),
                                     round(bestspotCoords[x,0]*scalefactor),:] 
                for x in range(bestspotCoords.shape[0])]).reshape(bestspotCoords.shape[0],3)
    bestdelta=np.sum(np.abs(bestIntensity-refIntensity))
    for eachx in np.linspace(-padratio*dev4search, padratio*dev4search, ninterval4search):
        for eachy in np.linspace(-padratio*dev4search, padratio*dev4search, ninterval4search):
            currentspotCoords=spotCoordsTIF.copy()
            currentratioX=padratio+eachx
            currentpadlengthX=min(min(round(tissuelength*currentratioX),spotMinXtif),wtif/scalefactor-spotCoordsTIF[:,0].max() ) #padding length
            newMinX=currentspotCoords[:,0].min()-currentpadlengthX # col
            newMaxX=currentspotCoords[:,0].max()+currentpadlengthX # col
            hnewMinX=round(newMinX*scalefactor)+0 #PIL Image cropping is 0-index and end-exclusive
            hnewMaxX=round(newMaxX*scalefactor)+1
            currentspotCoords[:,0]=currentspotCoords[:,0]-newMinX

            currentratioY=padratio+eachy
            currentpadlengthY=min(min(round(tissuelength*currentratioY),spotMinYtif),htif/scalefactor-spotCoordsTIF[:,1].max()) #padding length
            newMinY=currentspotCoords[:,1].min()-currentpadlengthY # row
            newMaxY=currentspotCoords[:,1].max()+currentpadlengthY # row
            hnewMinY=round(newMinY*scalefactor)+0
            hnewMaxY=round(newMaxY*scalefactor)+1
            currentspotCoords[:,1]=currentspotCoords[:,1]-newMinY
            
            hcropped=Image.fromarray((scale_imgIntensity.copy()* 255).astype(np.uint8)).convert('RGB').crop((hnewMinX,hnewMinY,hnewMaxX,hnewMaxY))
            hcropped=(np.asarray(hcropped)/255).astype(np.float32)
            currentIntensity=np.array([hcropped[round(currentspotCoords[x,1]*scalefactor),
                                     round(currentspotCoords[x,0]*scalefactor),:] 
                                     for x in range(currentspotCoords.shape[0])]).reshape(currentspotCoords.shape[0],3)

            currentdelta=np.sum(np.abs(currentIntensity-refIntensity))

            if currentdelta<bestdelta:
                bestdelta=currentdelta
                bestratioX=currentratioX                  
                bestratioY=currentratioY                  
    return [bestratioX,bestratioY]    

def optPadlength4(tissuelength, padratio, spotCoordsTIF,scale_imgIntensity, scalefactor, dev4search=0.2, ninterval4search=50):
    org_hires=Image.fromarray((scale_imgIntensity* 255).astype(np.uint8)).convert('RGB')
    refIntensity=np.array([scale_imgIntensity[round(spotCoordsTIF[x,1]*scalefactor),
                                     round(spotCoordsTIF[x,0]*scalefactor),:] 
                for x in range(spotCoordsTIF.shape[0])]).reshape(spotCoordsTIF.shape[0],3)
    htif,wtif,ncolor=scale_imgIntensity.shape

    spotMinXtif=spotCoordsTIF[:,0].min()
    spotMinYtif=spotCoordsTIF[:,1].min()
    bestspotCoords=spotCoordsTIF.copy()

    bestratioX=padratio
    bestpadlengthX=round(tissuelength*bestratioX) #padding length
    newMinX=bestspotCoords[:,0].min()-bestpadlengthX #col/width/left
    #newMaxX=bestspotCoords[:,0].max()+bestpadlengthX
    himgMinX=round(abs(newMinX)*scalefactor)+0
    hnew_width = round((bestspotCoords[:,0].max()-bestspotCoords[:,0].min()+1+2*bestpadlengthX)*scalefactor)
    bestspotCoords[:,0]=bestspotCoords[:,0]-newMinX

    bestratioY=padratio
    bestpadlengthY=round(tissuelength*bestratioY) 
    newMinY=bestspotCoords[:,1].min()-bestpadlengthY # row
    #newMaxY=bestspotCoords[:,1].max()+bestpadlengthY 
    himgMinY=round(abs(newMinY)*scalefactor)+0
    hnew_height = round((bestspotCoords[:,1].max()-bestspotCoords[:,1].min()+1+2*bestpadlengthY)*scalefactor)
    bestspotCoords[:,1]=bestspotCoords[:,1]-newMinY
    
    hpadded=Image.new("RGB", (hnew_width, hnew_height), (0, 0, 0))
    hpadded.paste(org_hires,(himgMinX,himgMinY))
    hpadded=(np.asarray(hpadded)/255).astype(np.float32)
    #print(hpadded.shape)
    #print([round(bestspotCoords[:,1].max()*scalefactor),round(bestspotCoords[:,0].max()*scalefactor)])

    bestIntensity=np.array([hpadded[round(bestspotCoords[x,1]*scalefactor),
                                     round(bestspotCoords[x,0]*scalefactor),:] 
                for x in range(bestspotCoords.shape[0])]).reshape(bestspotCoords.shape[0],3)
    bestdelta=np.sum(np.abs(bestIntensity-refIntensity))
    for eachx in np.linspace(-padratio*dev4search, padratio*dev4search, ninterval4search):
        for eachy in np.linspace(-padratio*dev4search, padratio*dev4search, ninterval4search):
            currentspotCoords=spotCoordsTIF.copy()
            
            currentratioX=padratio+eachx
            currentpadlengthX=round(tissuelength*currentratioX)  #padding length
            newMinX=currentspotCoords[:,0].min()-currentpadlengthX # col
            #newMaxX=currentspotCoords[:,0].max()+currentpadlengthX # col
            himgMinX=round(abs(newMinX)*scalefactor)+0
            hnew_width = round((currentspotCoords[:,0].max()-currentspotCoords[:,0].min()+1+2*currentpadlengthX)*scalefactor)
            currentspotCoords[:,0]=currentspotCoords[:,0]-newMinX

            currentratioY=padratio+eachy
            currentpadlengthY=round(tissuelength*currentratioY) 
            newMinY=currentspotCoords[:,1].min()-currentpadlengthY # row
            #newMaxY=currentspotCoords[:,1].max()+currentpadlengthY # row
            himgMinY=round(abs(newMinY)*scalefactor)+0
            hnew_height = round((currentspotCoords[:,1].max()-currentspotCoords[:,1].min()+1+2*currentpadlengthY)*scalefactor)
            currentspotCoords[:,1]=currentspotCoords[:,1]-newMinY
            
            hpadded=Image.new("RGB", (hnew_width, hnew_height), (0, 0, 0))
            hpadded.paste(org_hires,(himgMinX,himgMinY))
            hpadded=(np.asarray(hpadded)/255).astype(np.float32)
            currentIntensity=np.array([hpadded[round(currentspotCoords[x,1]*scalefactor),
                                     round(currentspotCoords[x,0]*scalefactor),:] 
                                     for x in range(currentspotCoords.shape[0])]).reshape(currentspotCoords.shape[0],3)

            currentdelta=np.sum(np.abs(currentIntensity-refIntensity))

            if currentdelta<bestdelta:
                bestdelta=currentdelta
                bestratioX=currentratioX                  
                bestratioY=currentratioY     
    return [bestratioX,bestratioY]    


def padAdata(adata,spatial_key,library_id, padratio=0.1, doMask=False, withOuts=False,dev4search=0.2,ninterval4search=50):
    adata_pad = adata.copy()
    newobsm=adata_pad.obsm.copy() # spot coords at full TIFF resolution
    newobs=adata_pad.obs.copy()
    newobsm[spatial_key]=newobsm[spatial_key][list(np.where(newobs['in_tissue'].to_numpy()==1)[0])] # new obsm['spatial']
    newobs=newobs.loc[newobs['in_tissue']==1] #new obs
    scale_imgIntensity=adata_pad.uns[spatial_key][library_id]['images']['hires']
    htif,wtif,ncolor=scale_imgIntensity.shape

    scalelow=scalehigh=1
    if 'tissue_lowres_scalef' in adata_pad.uns[spatial_key][library_id]['scalefactors'].keys():
        scalelow=adata_pad.uns[spatial_key][library_id]['scalefactors']['tissue_lowres_scalef']
    if 'tissue_hires_scalef' in adata_pad.uns[spatial_key][library_id]['scalefactors'].keys():
        scalehigh=adata_pad.uns[spatial_key][library_id]['scalefactors']['tissue_hires_scalef']

    xy_indices=newobsm[spatial_key].copy()
    intissueMinx=(xy_indices[:,0].min())
    intissueMaxx=(xy_indices[:,0].max())
    intissueMiny=(xy_indices[:,1].min())
    intissueMaxy=(xy_indices[:,1].max())

    #@@ change to use tissue boundary not just in_tissue spot boundary to decide padding
    deltax=xy_indices[:,0].max()-xy_indices[:,0].min()
    deltay=xy_indices[:,1].max()-xy_indices[:,1].min()
    tissuelength=max(deltax,deltay)
    #padlength=round(tissuelength*padratio) #padding length

    bestratioX, bestratioY=optPadlength4(tissuelength=tissuelength, padratio=padratio,spotCoordsTIF=xy_indices, 
                                         scale_imgIntensity=scale_imgIntensity,
                                        scalefactor=scalehigh, dev4search=dev4search, ninterval4search=ninterval4search)
    print((bestratioX, bestratioY))
    bestpadlengthX=round(tissuelength*bestratioX) 
    bestpadlengthY=round(tissuelength*bestratioY) 


    newMinX=xy_indices[:,0].min()-bestpadlengthX # col/width/left
    #newMaxX=xy_indices[:,0].max()+bestpadlengthX
    newMinY=xy_indices[:,1].min()-bestpadlengthY # row/height/top
    #newMaxY=xy_indices[:,1].max()+bestpadlengthY


    
    #shift spot coords to match newly cropped img
    newobsm[spatial_key][:,0]=newobsm[spatial_key][:,0]-newMinX
    newobsm[spatial_key][:,1]=newobsm[spatial_key][:,1]-newMinY

    # new TIFF start X = newMinX
    # new TIFF start Y = newMinY    
    #convert to new starts in actual hires or lowres images
    himgMinX=round(abs(newMinX)*scalehigh)+0
    himgMinY=round(abs(newMinY)*scalehigh)+0
    
    limgMinX=round(abs(newMinX)*scalelow)+0
    limgMinY=round(abs(newMinY)*scalelow)+0
  
    hnew_width = round((xy_indices[:,0].max()-xy_indices[:,0].min()+1+2*bestpadlengthX)*scalehigh)
    hnew_height = round((xy_indices[:,1].max()-xy_indices[:,1].min()+1+2*bestpadlengthY)*scalehigh)
    lnew_width = round((xy_indices[:,0].max()-xy_indices[:,0].min()+1+2*bestpadlengthX)*scalelow)
    lnew_height = round((xy_indices[:,1].max()-xy_indices[:,1].min()+1+2*bestpadlengthY)*scalelow)
    #hnew_width = hnewMaxX-hnewMinX + 1
    #hnew_height = hnewMaxY-hnewMinY + 1
    #lnew_width = lnewMaxX-lnewMinX + 1
    #lnew_height = lnewMaxY-lnewMinY + 1

    #left, top, right, bottom
    if doMask:
        if 'tissue_lowres_scalef' in adata_pad.uns[spatial_key][library_id]['scalefactors'].keys():
            masked_lowres=maskNonTissue(adata_pad.uns[spatial_key][library_id]['images']['lowres'],intissueMinx,intissueMiny,intissueMaxx,intissueMaxy,None)
            masked_lowres=Image.fromarray((masked_lowres*255).astype(np.uint8)).convert('RGB')
            lpadded=Image.new("RGB", (lnew_width, lnew_height), (0, 0, 0))
            lpadded.paste(masked_lowres,(limgMinX,limgMinY))
        masked_hires=maskNonTissue(adata_pad.uns[spatial_key][library_id]['images']['hires'],intissueMinx,intissueMiny,intissueMaxx,intissueMaxy,None)
        masked_hires=Image.fromarray((masked_hires*255).astype(np.uint8)).convert('RGB')
        hpadded=Image.new("RGB", (hnew_width, hnew_height), (0, 0, 0))
        hpadded.paste(masked_hires,(himgMinX,himgMinY))
    else:
        if 'tissue_lowres_scalef' in adata_pad.uns[spatial_key][library_id]['scalefactors'].keys():
            org_lowres=Image.fromarray((adata_pad.uns[spatial_key][library_id]['images']['lowres'] * 255).astype(np.uint8)).convert('RGB')
            lpadded=Image.new("RGB", (lnew_width, lnew_height), (0, 0, 0))
            Image.Image.paste(lpadded,org_lowres,(limgMinX,limgMinY))
        org_hires=Image.fromarray((adata_pad.uns[spatial_key][library_id]['images']['hires'] * 255).astype(np.uint8)).convert('RGB')
        hpadded=Image.new("RGB", (hnew_width, hnew_height), (0, 0, 0))
        hpadded.paste(org_hires,(himgMinX,himgMinY))

    if 'tissue_lowres_scalef' in adata_pad.uns[spatial_key][library_id]['scalefactors'].keys():
        adata_pad.uns[spatial_key][library_id]['images']['lowres']=(np.asarray(lpadded)/255).astype(np.float32)
    adata_pad.uns[spatial_key][library_id]['images']['hires']=(np.asarray(hpadded)/255).astype(np.float32)
    adata_pad.obs=newobs
    adata_pad.obsm=newobsm
    if withOuts:
        return [adata_pad, {'padlength':padlength,'scalehigh':scalehigh,'tissuelength':tissuelength,'hnew_width':hnew_width,'hnew_height':hnew_height,
                        'himgMinX':himgMinX, 'himgMinY':himgMinY,
                       'newMinX':newMinX,'newMaxX':newMaxX,'newMinY':newMinY,'newMaxY':newMaxY}]
    else:
        return adata_pad

def maskNonTissue(imgarray,minx,miny,maxx,maxy,scaling=None):
    test=imgarray.copy() #full image before cropping
    mask=standard_mask_fn(Image.fromarray((test* 255).astype(np.uint8)).convert('RGB'),2000)
    test[~np.array(mask)]=0
    if scaling is not None:
        minx=round(minx*scaling)
        miny=round(miny*scaling)
        maxx=round(maxx*scaling)
        maxy=round(maxy*scaling)
        test[0:minx,:,:]=0
        test[(maxx+1):-1,:,:]=0
        test[:,0:miny,:]=0
        test[:,(maxy+1):-1,:]=0
    return test

    
def encodeLBL(lbl_map,classes=None):
    elbl_map=np.zeros((lbl_map.shape[0],lbl_map.shape[1],np.unique(lbl_map).size))
    if classes is not None:
        classes=np.sort(np.unique(lbl_map)).astype(np.float64).tolist()
    for each in range(len(classes)):
        elbl_map[lbl_map==classes[each],each]=1
    return elbl_map


def arr2Img(imgarray):
    if imgarray.max()<=1:
        imgarray=(imgarray*255)
    return Image.fromarray(imgarray.astype(np.uint8)).convert("RGB")



    
def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    smooth = 0.0001
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def dice_coef_multilabel(y_true, y_pred, numLabels):
    dice=0
    for index in range(numLabels):
        dice += dice_coef(y_true[:,:,:,index], y_pred[:,:,:,index])
    return dice/numLabels # taking average

from keras import backend as K
def iou_coef(y_true, y_pred, smooth=1):
  intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
  union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
  iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
  return iou


def cropAdataByHires(adata,bestpadlengthX,bestpadlengthY, finalhires=None):
    adata_cp=adata.copy()
    spatial_key=list(adata_cp.uns.keys())[0]
    library_key=list(adata_cp.uns[spatial_key].keys())[0]
    scalehigh=adata_cp.uns[spatial_key][library_key]['scalefactors']['tissue_'+str('hires')+'_scalef']        
    himgX=adata_cp.uns[spatial_key][library_key]['images']['hires'].shape[0]
    himgY=adata_cp.uns[spatial_key][library_key]['images']['hires'].shape[1]
    limgX=adata_cp.uns[spatial_key][library_key]['images']['hires'].shape[0]
    limgY=adata_cp.uns[spatial_key][library_key]['images']['hires'].shape[1]
    
    xy_indices=adata_cp.obsm['spatial'].copy()
    x0=xy_indices[:,0].min()
    y0=xy_indices[:,1].min()
    xM=xy_indices[:,0].max()
    yM=xy_indices[:,1].max()        

    newMinX=x0-bestpadlengthX # col?
    newMinY=y0-bestpadlengthY # row?
    newMaxX=xM+bestpadlengthX
    newMaxY=yM+bestpadlengthY
    
    if finalhires is not None:
        if round(newMaxX*scalehigh)+1>round(newMinX*scalehigh)+finalhires:
            print("ERROR: image width is too large to fit finalhires without cropping out tissue")
            return None

    #shift spot coords to match newly cropped img
    xy_indices[:,0]=xy_indices[:,0]-newMinX # = padlength
    xy_indices[:,1]=xy_indices[:,1]-newMinY
    hnewMinX=round(newMinX*scalehigh)+0 #PIL Image cropping is 0-index and end-exclusive
    hnewMinY=round(newMinY*scalehigh)+0
    hnewMaxX=round(newMaxX*scalehigh)+1
    hnewMaxY=round(newMaxY*scalehigh)+1
    if finalhires is not None:
        #round(newMaxX*scalehigh)+1-round(newMinX*scalehigh)=finalhires
        hnewMaxX=round(newMinX*scalehigh)+finalhires
        hnewMaxY=round(newMinY*scalehigh)+finalhires
    hcropped=Image.fromarray((adata_cp.uns[spatial_key][library_key]['images']['hires']* 255).astype(np.uint8)).convert('RGB').crop((hnewMinX,hnewMinY,hnewMaxX,hnewMaxY))
    adata_cp.uns[spatial_key][library_key]['images']['hires']=(np.asarray(hcropped)/255).astype(np.float32)

    if 'tissue_'+str('lowres')+'_scalef' in list(adata_cp.uns[spatial_key][library_key]['scalefactors'].keys()):
        scalelow=adata_cp.uns[spatial_key][library_key]['scalefactors']['tissue_'+str('lowres')+'_scalef']
        lnewMinX=round(newMinX*scalelow)+0 #PIL Image cropping is 0-index and end-exclusive
        lnewMaxX=round(newMaxX*scalelow)+1
        lnewMinY=round(newMinY*scalelow)+0
        lnewMaxY=round(newMaxY*scalelow)+1
        if finalhires is not None:
            lnewMaxX=round(newMinX*scalelow)+round(finalhires*scalelow/scalehigh)
            lnewMaxY=round(newMinY*scalelow)+round(finalhires*scalelow/scalehigh)
        lcropped=Image.fromarray((adata_cp.uns[spatial_key][library_key]['images']['lowres']* 255).astype(np.uint8)).convert('RGB').crop((lnewMinX,lnewMinY,lnewMaxX,lnewMaxY))
        adata_cp.uns[spatial_key][library_key]['images']['lowres']=(np.asarray(lcropped)/255).astype(np.float32)

    adata_cp.obs=adata_cp.obs.loc[adata_cp.obs['in_tissue']==1] #new obs
    adata_cp.obsm['spatial']=xy_indices
    return adata_cp

def padAdataByHires(adata,bestpadlengthX,bestpadlengthY, finalhires=None):
    adata_cp=adata.copy()
    spatial_key=list(adata_cp.uns.keys())[0]
    library_key=list(adata_cp.uns[spatial_key].keys())[0]
    scalehigh=adata_cp.uns[spatial_key][library_key]['scalefactors']['tissue_'+str('hires')+'_scalef']   

    xy_indices=adata_cp.obsm['spatial'].copy()
    padX4FinalHires=0
    padY4FinalHires=0
    if finalhires is not None:
            padX4FinalHires=(finalhires/scalehigh-(xy_indices[:,0].max()-xy_indices[:,0].min()+1))/2
            padY4FinalHires=(finalhires/scalehigh-(xy_indices[:,1].max()-xy_indices[:,1].min()+1))/2
    bestpadlengthX=max(padX4FinalHires, bestpadlengthX)#padding length
    bestpadlengthY=max(padY4FinalHires, bestpadlengthY) #padding length
        
    x0=xy_indices[:,0].min()
    y0=xy_indices[:,1].min()
    xM=xy_indices[:,0].max()
    yM=xy_indices[:,1].max()        

    newMinX=x0-bestpadlengthX # col?
    newMinY=y0-bestpadlengthY # row?
    
    #shift spot coords to match newly cropped img
    xy_indices[:,0]=xy_indices[:,0]-newMinX
    xy_indices[:,1]=xy_indices[:,1]-newMinY
    
    himgMinX=round(abs(newMinX)*scalehigh)+0
    himgMinY=round(abs(newMinY)*scalehigh)+0
    hnew_width = round((xy_indices[:,0].max()-xy_indices[:,0].min()+1+2*bestpadlengthX)*scalehigh)
    hnew_height = round((xy_indices[:,1].max()-xy_indices[:,1].min()+1+2*bestpadlengthY)*scalehigh)
    if finalhires is not None:
        hnew_width=max(hnew_width,finalhires)
        hnew_height=max(hnew_height,finalhires)

    hpadded=Image.new("RGB", (hnew_width, hnew_height), (0, 0, 0))
    orgimg=Image.fromarray((adata_cp.uns[spatial_key][library_key]['images']['hires'] * 255).astype(np.uint8)).convert('RGB')
    hpadded.paste(orgimg,(himgMinX,himgMinY))
    adata_cp.uns[spatial_key][library_key]['images']['hires']=(np.asarray(hpadded)/255).astype(np.float32)

    if 'tissue_'+str('lowres')+'_scalef' in list(adata_cp.uns[spatial_key][library_key]['scalefactors'].keys()):
        scalelow=adata_cp.uns[spatial_key][library_key]['scalefactors']['tissue_'+str('lowres')+'_scalef']
        limgMinX=round(abs(newMinX)*scalelow)+0
        limgMinY=round(abs(newMinY)*scalelow)+0  
        lnew_width = round((xy_indices[:,0].max()-xy_indices[:,0].min()+1+2*bestpadlengthX)*scalelow)
        lnew_height = round((xy_indices[:,1].max()-xy_indices[:,1].min()+1+2*bestpadlengthY)*scalelow)
        if finalhires is not None:
            lnew_width=max(lnew_width,round(finalhires*scalelow/scalehigh))
            lnew_height=max(lnew_height,round(finalhires*scalelow/scalehigh))
        lpadded=Image.new("RGB", (lnew_width, lnew_height), (0, 0, 0))
        orgimg=Image.fromarray((adata_cp.uns[spatial_key][library_key]['images']['lowres'] * 255).astype(np.uint8)).convert('RGB')
        lpadded.paste(orgimg,(limgMinX,limgMinY))
        adata_cp.uns[spatial_key][library_key]['images']['lowres']=(np.asarray(lpadded)/255).astype(np.float32)

    adata_cp.obs=adata_cp.obs.loc[adata_cp.obs['in_tissue']==1] #new obs
    adata_cp.obsm['spatial']=xy_indices
    return adata_cp


def prep4AF(im_mov, mask_mov=None,max_reg_dim=512,defaultvalue=0):
    #@@ simplify to extract mask_mov/fix from the input img directly
    mask_fn = lambda pil_im: standard_mask_fn(pil_im=pil_im,max_mask_dim=max_reg_dim)    
    # Preprocess and verify the images.
    if not isinstance(im_mov, Image.Image): im_mov = Image.fromarray(im_mov)
    #if not isinstance(im_mov, Image.Image): im_fix = Image.fromarray(im_fix)
    #assert np.array_equal(im_mov.size, im_fix.size), 'Anndata objects must reference images of the same size.'
    # Preprocess and verify the masks if a masking function present.
    if mask_mov is not None:
        if not isinstance(mask_mov, Image.Image): mask_mov = Image.fromarray(mask_mov)
    elif mask_fn is not None: mask_mov = mask_fn(im_mov)
    assert np.array_equal(im_mov.size, mask_mov.size) and mask_mov.mode == '1', \
            'mask_mov must be a binary PIL mask with the same size as im_mov.'
    #if mask_fix is not None:
    #    if not isinstance(mask_fix, Image.Image): mask_fix = Image.fromarray(mask_fix)
    #elif mask_fn is not None: mask_fix = mask_fn(im_fix)
    #assert np.array_equal(im_fix.size, mask_fix.size) and mask_fix.mode == '1', \
    #        'mask_fix must be a binary PIL mask with the same size as im_fix.'
    # APPLY MASKS TO INITIAL IMAGES
    im_mov = np.array(im_mov)
    im_mov[~np.asarray(mask_mov).astype(bool)] = defaultvalue
    im_mov = Image.fromarray(im_mov)
    return im_mov, mask_mov

def afTransform(im_mov, im_fix, mask_mov, mask_fix,spatial_mov,max_reg_dim=512,defaultvalue=0, search_factor=5 ,use_principal_axis=True):
    
    # make a "small" size for faster af
    im_mov_r, im_fix_r, mask_mov_r, mask_fix_r = map(lambda x, resample: resize_to_max_dim_pil(x,
                                                                                                max_dim=max_reg_dim,
                                                                                                resample=resample),
                                                        (im_mov, im_fix, mask_mov, mask_fix),
                                                        (Image.Resampling.BILINEAR,
                                                        Image.Resampling.BILINEAR,
                                                        Image.Resampling.NEAREST,
                                                        Image.Resampling.NEAREST))
        
    # perform affine registration; omit itk for now.
    im_mov_ra, mask_mov_ra, aff_tfm_info = ants_align_pil2(mov_pil=im_mov_r,
                                                              fix_pil=im_fix_r,
                                                              mov_mask_pil=mask_mov_r,
                                                              defaultvalue=defaultvalue,
                                                              type_of_transform='Affine',
                                                              search_factor=search_factor,
                                                              use_principal_axis=use_principal_axis
                                                          )

    # upscale the affine transform
    large_aff_tfm = ants.read_transform(aff_tfm_info['fwdtransforms'][0]) # assume only one matrix out.
    large_aff_tfm = scale_affine(trf=large_aff_tfm,
                                        orig_shape_2d=im_mov_r.size[::-1],
                                        new_shape_2d=im_mov.size[::-1])
    large_aff_tfm_itk = ants2itk_affine(large_aff_tfm)

    # align the large images
    im_mov_a, mask_mov_a = map(lambda input, interp, dv: apply_itk_trf_image(
                input=input,
                trf=large_aff_tfm_itk,
                interpolator=interp,
                defaultPixelValue=dv,
                outputPixelType=sitk.sitkUnknown,
                useNearestNeighborExtrapolator=True),
                                    (np.asarray(im_mov).astype(float),
                                        np.asarray(mask_mov.convert('RGB')).astype(float)),
                                    (sitk.sitkLinear, sitk.sitkNearestNeighbor),
                                    (float(defaultvalue), 0.0))

    inv_large_aff_tfm_itk=large_aff_tfm_itk.GetInverse()
    # need to apply inverse transform to mimic image's resample.
    spatial_mov_a = register_spatial_with_itk_points(
                    spatial_data=spatial_mov,
                    inverse_itk_trf=inv_large_aff_tfm_itk, 
                    spatial_data_indexing='xy')
    return im_mov_a, mask_mov_a, spatial_mov_a

   
def scaleAdata(adataset,  ratio=0.07, finalhires=512, coarse=0, onlyCropPad=False, search_factor=20, use_principal_axis=False):
    res='hires'
    rescaled=[]
    spatial_key=list(adataset[0].uns.keys())[0]
    library_key=list(adataset[0].uns[spatial_key].keys())[0]
    
    if not onlyCropPad:
     af=[]
     for each in range(len(adataset)):
        tmp=adataset[each].copy()
        spatial_mov=tmp.uns[spatial_key].values()
        movd_spatial_nm = next(iter(tmp.uns[spatial_key].keys()))
        im_mov=Image.fromarray((tmp.uns[spatial_key][movd_spatial_nm]['images'][res] * 255).astype(np.uint8)).convert('RGB')
        scalefactor_mov = tmp.uns[spatial_key][movd_spatial_nm]['scalefactors']['tissue_%s_scalef' % res]
        spatial_mov_scaled = tmp.obsm['spatial'] * scalefactor_mov
        [im_mov, mask_mov]=prep4AF(im_mov, mask_mov=None)
        tmp.uns['spatial'][movd_spatial_nm]['images'][res]=im_mov
        tmp.uns['spatial'][movd_spatial_nm]['images']['%s_mask' % res]=mask_mov
        tmp.obsm['spatial'] =spatial_mov_scaled
        af.append(tmp)
        
     rest=set(range(len(adataset))).difference(set([coarse]))
     im_fix=af[coarse].uns[spatial_key][next(iter(af[coarse].uns[spatial_key].keys()))]['images'][res]
     mask_fix=af[coarse].uns[spatial_key][next(iter(af[coarse].uns[spatial_key].keys()))]['images']['%s_mask' % res]
     for each in rest:
        movd_spatial_nm=next(iter(af[each].uns[spatial_key].keys()))
        im_mov=af[each].uns[spatial_key][movd_spatial_nm]['images'][res]
        mask_mov=af[each].uns[spatial_key][movd_spatial_nm]['images']['%s_mask' % res]
        spatial_mov=af[each].obsm['spatial'] 
        [im_mov_a,mask_mov_a, spatial_mov_a]=afTransform(im_mov, im_fix, mask_mov, mask_fix,spatial_mov,max_reg_dim=256, search_factor=search_factor,
                                                       use_principal_axis=use_principal_axis)
        af[each].uns['spatial'][movd_spatial_nm]['images'][res]=im_mov_a
        af[each].obsm['spatial']=spatial_mov_a
        
     for each in range(len(af)):
        movd_spatial_nm=next(iter(af[each].uns[spatial_key].keys()))
        af[each].uns[spatial_key][movd_spatial_nm]['scalefactors']['tissue_%s_scalef' % res]=1
        af[each].uns['spatial'][movd_spatial_nm]['images'][res] = np.asarray(af[each].uns['spatial'][movd_spatial_nm]['images'][res]) / 255
        af[each].obsm['spatial']= af[each].obsm['spatial'] #/ scalefactor_mov
        del af[each].uns[spatial_key][movd_spatial_nm]['images']['%s_mask' % res] #remove  mask to be sure 
    
    #first scaling
     rangeX=np.median([each.obsm['spatial'][:,0].max()-each.obsm['spatial'][:,0].min()+1 for each in af] )
     rangeY=np.median([each.obsm['spatial'][:,1].max()-each.obsm['spatial'][:,1].min()+1 for each in af] )
     scalings=[min(rangeX/(each.obsm['spatial'][:,0].max()-each.obsm['spatial'][:,0].min()+1),rangeY/(each.obsm['spatial'][:,1].max()-each.obsm['spatial'][:,1].min()+1)) for each in af]
    
    
     for each in range(len(af)):
        tmp=af[each].copy()
        movd_spatial_nm = next(iter(tmp.uns[spatial_key].keys()))
        img=Image.fromarray((af[each].uns[spatial_key][movd_spatial_nm]['images'][res]*255).astype(np.uint8))
        img=img.resize([round(dim*scalings[each]) for dim in img.size])
        imgarray=np.array(img)/255.0
        tmp.uns[spatial_key][library_key]['images'][res]=imgarray
        tmp.obsm['spatial']=scalings[each]*tmp.obsm['spatial']
        rescaled.append(tmp)
        
     print("rangeX="+str(rangeX))
     print([each.obsm['spatial'][:,0].max()-each.obsm['spatial'][:,0].min()+1 for each in af])
     print([each.obsm['spatial'][:,0].max()-each.obsm['spatial'][:,0].min()+1 for each in rescaled])
     print("rangeY="+str(rangeY))
     print([each.obsm['spatial'][:,1].max()-each.obsm['spatial'][:,1].min()+1 for each in af])
     print([each.obsm['spatial'][:,1].max()-each.obsm['spatial'][:,1].min()+1 for each in rescaled])
    else: # if only need to crop and pad to reach final resolution
     rescaled=[x.copy() for x in adataset]   

    # update ranges and tissuelength with rescaled 
    rangeX=np.median([each.obsm['spatial'][:,0].max()-each.obsm['spatial'][:,0].min()+1 for each in rescaled] )
    rangeY=np.median([each.obsm['spatial'][:,1].max()-each.obsm['spatial'][:,1].min()+1 for each in rescaled] )
    tissuelength=min(rangeX, rangeY)

    # use padding to center the spots 
    centerpadlengthXs=[]
    centerpadlengthYs=[]
    for each in range(len(rescaled)):
        htif,wtif,ncolor=rescaled[each].uns[spatial_key][library_key]['images'][res].shape
        xy_indices=rescaled[each].obsm['spatial'].copy()
        scalehigh=rescaled[each].uns[spatial_key][library_key]['scalefactors']['tissue_'+str(res)+'_scalef']
        bestpadlengthX= np.mean([xy_indices[:,0].min(), wtif/scalehigh-xy_indices[:,0].max()])#padding length
        bestpadlengthY= np.mean([xy_indices[:,1].min(), htif/scalehigh-xy_indices[:,1].max()]) #padding length
        centerpadlengthXs.append(bestpadlengthX)
        centerpadlengthYs.append(bestpadlengthY)
    centerpadlengthX=np.max(centerpadlengthXs)
    centerpadlengthY=np.max(centerpadlengthYs)
    for each in range(len(rescaled)):
        rescaled[each]=padAdataByHires(rescaled[each],centerpadlengthX,centerpadlengthY, finalhires=round(finalhires*1.25))
    
    # step 4
    padlengthXs=[]
    padlengthYs=[]
    for each in range(len(rescaled)):
        htif,wtif,ncolor=rescaled[each].uns[spatial_key][library_key]['images'][res].shape
        scalehigh=rescaled[each].uns[spatial_key][library_key]['scalefactors']['tissue_'+str(res)+'_scalef']
        xy_indices=rescaled[each].obsm['spatial'].copy()
        x_min_bytissue=np.where(rescaled[each].uns[spatial_key][library_key]['images']['hires'].sum(axis=2)>0)[1].min()/scalehigh
        x_max_bytissue=np.where(rescaled[each].uns[spatial_key][library_key]['images']['hires'].sum(axis=2)>0)[1].max()/scalehigh
        y_min_bytissue=np.where(rescaled[each].uns[spatial_key][library_key]['images']['hires'].sum(axis=2)>0)[0].min()/scalehigh
        y_max_bytissue=np.where(rescaled[each].uns[spatial_key][library_key]['images']['hires'].sum(axis=2)>0)[0].max()/scalehigh
        bestpadlengthX=max(max(round(tissuelength*ratio),xy_indices[:,0].min()-x_min_bytissue+5), (x_max_bytissue-xy_indices[:,0].max()+5))
        bestpadlengthY=max(max(round(tissuelength*ratio),xy_indices[:,1].min()-y_min_bytissue+5), (y_max_bytissue-xy_indices[:,1].max()+5)) #padding length
        padlengthXs.append(bestpadlengthX)
        padlengthYs.append(bestpadlengthY)
    bestpadlengthX=np.max(padlengthXs)
    bestpadlengthY=np.max(padlengthYs)
    
    shrinks=[]
    for each in range(len(rescaled)):
        scalehigh=rescaled[each].uns[spatial_key][library_key]['scalefactors']['tissue_'+str(res)+'_scalef']
        xy_indices=rescaled[each].obsm['spatial'].copy()
        newMinX=xy_indices[:,0].min()-bestpadlengthX # col?
        newMinY=xy_indices[:,1].min()-bestpadlengthY # row?
        newMaxX=xy_indices[:,0].max()+bestpadlengthX
        newMaxY=xy_indices[:,1].max()+bestpadlengthY
        shrink=min(1,(finalhires-1)*1.0/(round(newMaxX*scalehigh)-round(newMinX*scalehigh)))
        shrinks.append(shrink)
    bestshrink=np.min(shrinks)
    bestpadlengthX=bestpadlengthX*bestshrink
    bestpadlengthY=bestpadlengthY*bestshrink
    
    rescaled2=[]
    for each in range(len(rescaled)):
        tmp=rescaled[each].copy()
        img=Image.fromarray((rescaled[each].uns[spatial_key][library_key]['images'][res]*255).astype(np.uint8))
        img=img.resize([round(dim*bestshrink) for dim in img.size])
        imgarray=np.array(img)/255.0
        tmp.uns[spatial_key][library_key]['images'][res]=imgarray
        tmp.obsm['spatial']=bestshrink*tmp.obsm['spatial']
        rescaled2.append(tmp)
    rescaled=rescaled2
    
    print([bestpadlengthX,bestpadlengthY])
    print([each.obsm['spatial'][:,0].min() for each in rescaled])
    print([each.obsm['spatial'][:,1].min() for each in rescaled])
    
    for each in range(len(rescaled)):
        rescaled[each]=cropAdataByHires(rescaled[each],bestpadlengthX,bestpadlengthY, finalhires=finalhires)
    return rescaled

def scale_spatial_coords2(X, ref_min=None, ref_max=None, max_val=10.0):
    if ref_min is None:
        ref_min = X.min(0)
    X = X - ref_min
    if ref_max is None:
        ref_max = X.max(0)
    X = X / ref_max
    return X * max_val

def createGrid(xDim=512, yDim=512):
 grid=[]
 for x in range(xDim): #adata_warped.uns[spatial_key][library_key]['images']['hires'].shape[1]
    for y in range(yDim):
        grid.append([x,y])
 grid=np.array(grid)
 return grid


def deffPixel(rs, pixel_org):
    pixel_a=register_spatial_with_itk_points(pixel_org,
                                         rs[1]['affine']['inv_deff'],
                                         spatial_data_indexing='ij')
    pixel_d=register_spatial_with_def_field_points(
                    spatial_data=pixel_a,
                    inverse_def_field=rs[1]['dense']['inv_deff'],
                    spatial_data_indexing='xy',
                    inverse_def_field_indexing='ij')    
    return pixel_d


def iterModel(adata_unwarped, adata_warped, model_sm, resolution=512, maxRound=4, maxratio=0.01, only1round=False, search_factor=5):
  spatial_key=list(adata_unwarped.uns.keys())[0]
  library_id=list(adata_unwarped.uns[spatial_key].keys())[0]
    
  pixel_org=createGrid(xDim=adata_warped.uns[spatial_key][library_id]['images']['hires'].shape[1], yDim=adata_warped.uns[spatial_key][library_id]['images']['hires'].shape[0])

  new_img_iter1=[]
  new_spatial_iter1=[]
  new_mse_iter1=[]
  new_mse=[]
  new_spatial=[]
  new_img=[]
    #first round  
  registration_data_points = sc_register(adata_mov=adata_warped,
                     adata_fix=adata_unwarped,
                     spatial_strategy='points',
                     spatial_target='hires',
                     max_reg_dim=resolution,
                     mode='dense',
                     model=model_sm,
                     dense_moving_img_name='sm_semi_base_moving_img',
                     dense_fixed_img_name='sm_semi_base_fixed_img',
                     dense_deff_name='def_output',
                     dense_inv_deff_name='inv_def_output',
                     search_factor=search_factor,
                     use_principal_axis=False)
  
  if only1round:
    return registration_data_points
  pixel_d=deffPixel(registration_data_points, pixel_org)
    
  fit=scale_spatial_coords2(registration_data_points[0].obsm['spatial'],ref_min=adata_warped.obsm['spatial'].min(0),
                          ref_max=(adata_warped.obsm['spatial']-adata_warped.obsm['spatial'].min(0)).max(0))
    
  previous=registration_data_points[0].copy()
  previousfit=fit
  
  #2nd round
    
  registration_data_points = sc_register(adata_mov=previous,
                     adata_fix=adata_unwarped,
                     spatial_strategy='points',
                     max_reg_dim=resolution,
                     mode='dense',
                     model=model_sm,
                     dense_moving_img_name='sm_semi_base_moving_img',
                     dense_fixed_img_name='sm_semi_base_fixed_img',
                     dense_deff_name='def_output',
                     dense_inv_deff_name='inv_def_output',
                     spot_diameter_unscaled=1,
                     dense_normalize=True)
  pixel_d=deffPixel(registration_data_points, pixel_d)

  fit=scale_spatial_coords2(registration_data_points[0].obsm['spatial'],ref_min=adata_warped.obsm['spatial'].min(0),
                          ref_max=(adata_warped.obsm['spatial']-adata_warped.obsm['spatial'].min(0)).max(0))
  first_delta_fit=(((fit[:,0] - previousfit[:,0])** 2+(fit[:,1] - previousfit[:,1])** 2).mean())
  round=1
  while round <=maxRound:
   registration_data_points = sc_register(adata_mov=previous,
                     adata_fix=adata_unwarped,
                     spatial_strategy='points',
                     max_reg_dim=resolution,
                     mode='dense',
                     model=model_sm,
                     dense_moving_img_name='sm_semi_base_moving_img',
                     dense_fixed_img_name='sm_semi_base_fixed_img',
                     dense_deff_name='def_output',
                     dense_inv_deff_name='inv_def_output',
                     spot_diameter_unscaled=1,
                     dense_normalize=True)
   pixel_d=deffPixel(registration_data_points, pixel_d)

   round=round+1
   fit=scale_spatial_coords2(registration_data_points[0].obsm['spatial'],ref_min=adata_warped.obsm['spatial'].min(0),
                          ref_max=(adata_warped.obsm['spatial']-adata_warped.obsm['spatial'].min(0)).max(0))
   delta_fit=(((fit[:,0] - previousfit[:,0])** 2+(fit[:,1] - previousfit[:,1])** 2).mean())
   dd_fit=abs(delta_fit-first_delta_fit) 
   if (dd_fit/first_delta_fit)<maxratio:
        break
   first_delta_fit= delta_fit
   previous=registration_data_points[0].copy()
   previousfit= fit
   pixel_final=deffPixel(registration_data_points, pixel_d)
   registration_data_points[1]['pixel']=pixel_d
    
  return registration_data_points
