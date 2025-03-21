library(png)
library(matrixStats)
library(ggpubr)
library(ggplot2)

maincolor=c("red2","orange","green","royalblue","cyan","violet","aquamarine2","coral","gold","yellow","cornflowerblue","darksalmon","darkgoldenrod","chartreuse","darkorange","darkturquoise","pink","magenta","brown","tan","blueviolet","goldenrod","green4","greenyellow","wheat4","darkorange1","royalblue1","aquamarine4","orchid4");
badcolor=colors()[which(sapply(colors(),function(x){grep("(white|aliceroyalblue|cornsilk|azure|beige|bisque|blanchedalmond|black|gray|gainsboro|ivory|grey[90-100]|lightyellow|linen|oldlace|mintcream|seashell|papayawhip|snow|whitesmoke)",x,perl=T)})>0)]
colorbank=c("antiquewhite4",maincolor, colors()[!(colors() %in% c(badcolor,maincolor))]);
minicolorbank=c('cyan','red2','darkorange')

rootdir='../../submission/'

################## Figure 3
# 10 runs
unaligned=as.matrix(read.csv(paste0(rootdir,"/data/Fig3A_NCC_unaligned_.csv"),row.names=1))
stacker=as.matrix(read.csv(paste0(rootdir,"/data/Fig3A_NCC_stacker_.csv"),row.names=1))
affine=as.matrix(read.csv(paste0(rootdir,"/data/Fig3A_NCC_affine_.csv"),row.names=1))
ants=as.matrix(read.csv(paste0(rootdir,"/data/Fig3A_NCC_ants_.csv"),row.names=1))
wreg=as.matrix(read.csv(paste0(rootdir,"/data/Fig3A_NCC_wreg_.csv"),row.names=1))

tag='small'
#tag='medium'
#tag='large'
#tag='manual'
df=data.frame("method"=c("moving",'Affine','WSIreg','ANTs',"STaCker"),
              "NCC"=c(unaligned[tag,],mean(affine[tag,]), wreg[tag,],
                      mean(ants[tag,]),mean(stacker[tag,])),
              'SE'=c(0,sqrt(var(affine[tag,])/ncol(affine)), 0,
                     sqrt(var(ants[tag,])/ncol(ants)),
                     sqrt(var(stacker[tag,])/ncol(stacker))))
df$method=factor(df$method,level=c('moving','Affine','WSIreg','ANTs','STaCker'))

pdf(paste0(rootdir,'/figures/Fig3A_NCC_',tag,'.pdf'),width=1.45,height=0.9,pointsize = 20)
ggplot(df) +
  geom_bar( aes(x=method, y=NCC), stat="identity", color="black",fill="white", alpha=0.7,width=0.8) +
  geom_errorbar( aes(x=method, ymin=NCC-SE, ymax=NCC+SE), width=0.1, colour="black", alpha=0.5, size=0.5)+
  ylim(0.,1)+coord_cartesian(ylim=c(0.3,0.95))+
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black"),
        axis.title.x=element_blank(),axis.text.x=element_blank(), 
        axis.title.y=element_blank())
dev.off()

## 100 runs
unaligned=as.matrix(read.csv(paste0(rootdir,"/data/Fig3A_NCC_unaligned_.csv"),row.names=1))
stacker=as.matrix(read.csv(paste0(rootdir,"/data/Fig3A_NCC_stacker_n100.csv"),row.names=1))
affine=as.matrix(read.csv(paste0(rootdir,"/data/Fig3A_NCC_affine_n100.csv"),row.names=1))
ants=as.matrix(read.csv(paste0(rootdir,"/data/Fig3A_NCC_ants_n100.csv"),row.names=1))
wreg=as.matrix(read.csv(paste0(rootdir,"/data/Fig3A_NCC_wreg_.csv"),row.names=1))

tag='small'
#tag='medium'
#tag='large'
#tag='manual'
df=data.frame("method"=c("moving",'Affine','WSIreg','ANTs',"STaCker"),
              "NCC"=c(unaligned[tag,],mean(affine[tag,]), wreg[tag,],
                      mean(ants[tag,]),mean(stacker[tag,])),
              'SE'=c(0,sqrt(var(affine[tag,])/ncol(affine)), 0,
                     sqrt(var(ants[tag,])/ncol(ants)),
                     sqrt(var(stacker[tag,])/ncol(stacker))))
df$method=factor(df$method,level=c('moving','Affine','WSIreg','ANTs','STaCker'))

pdf(paste0(rootdir,'/figures/Fig3A_NCC_',tag,'_n100.pdf'),width=1.45,height=0.9,pointsize = 20)
ggplot(df) +
  geom_bar( aes(x=method, y=NCC), stat="identity", color="black",fill="white", alpha=0.7,width=0.8) +
  geom_errorbar( aes(x=method, ymin=NCC-SE, ymax=NCC+SE), width=0.1, colour="black", alpha=0.5, size=0.5)+
  ylim(0.,1)+coord_cartesian(ylim=c(0.3,0.95))+
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black"),
        axis.title.x=element_blank(),axis.text.x=element_blank(), 
        axis.title.y=element_blank())
dev.off()

# Fig3B
unaligned=as.matrix(read.csv(paste0(rootdir,"/data/Fig3B_NCC_unaligned_.csv"),row.names=1))
stacker=as.matrix(read.csv(paste0(rootdir,"/data/Fig3B_NCC_stacker_.csv"),row.names=1))
affine=as.matrix(read.csv(paste0(rootdir,"/data/Fig3B_NCC_affine_.csv"),row.names=1))
ants=as.matrix(read.csv(paste0(rootdir,"/data/Fig3B_NCC_ants_.csv"),row.names=1))
wreg=as.matrix(read.csv(paste0(rootdir,"/data/Fig3B_NCC_wreg_.csv"),row.names=1))
tag='small'
#tag='medium'
#tag='large'
#tag='manual'
df=data.frame("method"=c("moving",'Affine','WSIreg','ANTs',"STaCker"),
              "NCC"=c(unaligned[tag,],mean(affine[tag,]), wreg[tag,],
                      mean(ants[tag,]),mean(stacker[tag,])),
              'SE'=c(0,sqrt(var(affine[tag,])/ncol(affine)), 0,
                     sqrt(var(ants[tag,])/ncol(ants)),
                     sqrt(var(stacker[tag,])/ncol(stacker))))
df$method=factor(df$method,level=c('moving','Affine','WSIreg','ANTs','STaCker'))

pdf(paste0(rootdir,'/figures/Fig3B_NCC_',tag,'.pdf'),width=1.5,height=0.9,pointsize = 20)
ggplot(df) +
  geom_bar( aes(x=method, y=NCC), stat="identity", color="black",fill="white", alpha=0.7,width=0.8) +
  geom_errorbar( aes(x=method, ymin=NCC-SE, ymax=NCC+SE), width=0.1, colour="black", alpha=0.5, size=0.5)+
  ylim(0.,1)+coord_cartesian(ylim=c(0.42,0.63))+
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black"),
        axis.title.x=element_blank(),axis.text.x=element_blank(), 
        axis.title.y=element_blank())
dev.off()

## 100 runs
unaligned=as.matrix(read.csv(paste0(rootdir,"/data/Fig3B_NCC_unaligned_.csv"),row.names=1))
stacker=as.matrix(read.csv(paste0(rootdir,"/data/Fig3B_NCC_stacker_n100.csv"),row.names=1))
affine=as.matrix(read.csv(paste0(rootdir,"/data/Fig3B_NCC_affine_n100.csv"),row.names=1))
ants=as.matrix(read.csv(paste0(rootdir,"/data/Fig3B_NCC_ants_n100.csv"),row.names=1))
wreg=as.matrix(read.csv(paste0(rootdir,"/data/Fig3B_NCC_wreg_.csv"),row.names=1))
tag='small'
#tag='medium'
#tag='large'
#tag='manual'
df=data.frame("method"=c("moving",'Affine','WSIreg','ANTs',"STaCker"),
              "NCC"=c(unaligned[tag,],mean(affine[tag,]), wreg[tag,],
                      mean(ants[tag,]),mean(stacker[tag,])),
              'SE'=c(0,sqrt(var(affine[tag,])/ncol(affine)), 0,
                     sqrt(var(ants[tag,])/ncol(ants)),
                     sqrt(var(stacker[tag,])/ncol(stacker))))
df$method=factor(df$method,level=c('moving','Affine','WSIreg','ANTs','STaCker'))

pdf(paste0(rootdir,'/figures/Fig3B_NCC_',tag,'_n100.pdf'),width=1.5,height=0.9,pointsize = 20)
ggplot(df) +
  geom_bar( aes(x=method, y=NCC), stat="identity", color="black",fill="white", alpha=0.7,width=0.8) +
  geom_errorbar( aes(x=method, ymin=NCC-SE, ymax=NCC+SE), width=0.1, colour="black", alpha=0.5, size=0.5)+
  ylim(0.,1)+coord_cartesian(ylim=c(0.42,0.63))+
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black"),
        axis.title.x=element_blank(),axis.text.x=element_blank(), 
        axis.title.y=element_blank())
dev.off()


################## Figure 4
load(paste0(rootdir,'data/Fig4_metrics.rda'))

reference=as.matrix(read.csv(paste0(rootdir,"/data/Fig4_coords_ref_pixel.csv"),row.names=1))
img<-readPNG(paste0(rootdir,"/data/Fig4_ref_pixel.png"))
h<-dim(img)[1]
w<-dim(img)[2]
pdf(paste0(rootdir,"/figures/Fig4_ref_pixelWspot.pdf"), width=8, height=8)
par(mar=c(0,0,0,0), xpd=NA, mgp=c(0,0,0), oma=c(0,0,0,0), ann=F)
plot.new()
plot.window(0:1, 0:1)
usr=c(0,1,0,1)
rasterImage(img, usr[1], usr[3], usr[2], usr[4])
points(0,0, cex=1,pch=16, col=rgb(.9,.9,.9,.5))
points(reference[,"x"]/512,(512-reference[,"y"])/512, cex=0.8,pch=16, col=rgb(.3,.3,.3,.5))
dev.off()

for(d in c('low','med','high')){
  ref=fig4_coords_ref
  fit=fig4_coords_unaligned[[d]]
  maxy=max(max(ref[,2]),max(fit[,2]))
  pdf(paste0(rootdir,"figures/Fig4_spatial_",d,"_unaligned.pdf"),width=8,height=8,pointsize = 10)
  plot(ref[,1],maxy-ref[,2],pch=16,col="grey60",lwd=0.1,xlab="",ylab="")
  points(fit[,1],maxy-fit[,2],pch=4,col='blue',lwd=1.5,xlab="",ylab="")
  dev.off()
}

for (d in c('low','med','high')){
  ref=fig4_coords_ref
  fit=fig4_coords_stacker[[d]][['t0']]
  maxy=max(max(ref[,2]),max(fit[,2]))
  pdf(paste0(rootdir,"figures/Fig4_spatial_",d,"_stacker.pdf"),width=8,height=8,pointsize = 10)
  plot(ref[,1],maxy-ref[,2],pch=16,col="grey30",lwd=0.1,xlab="",ylab="")
  points(fit[,1],maxy-fit[,2],pch=4,col='red',lwd=1,xlab="",ylab="")
  dev.off()
}
for (d in c('low','med','high')){
  ref=fig4_coords_ref
  fit=fig4_coords_stu[[d]]
  maxy=max(max(ref[,2]),max(fit[,2]))
  pdf(paste0(rootdir,"figures/Fig4_spatial_",d,"_STUtility.pdf"),width=8,height=8,pointsize = 10)
  plot(ref[,1],maxy-ref[,2],pch=16,col="grey40",lwd=0.1,xlab="",ylab="")
  points(fit[,1],maxy-fit[,2],pch=4,col='red',lwd=1,xlab="",ylab="")
  dev.off()
}

for (d in c('low','med','high')){
  ref=fig4_coords_ref
  fit=fig4_coords_paste[[d]]
  maxy=max(max(ref[,2]),max(fit[,2]))
  pdf(paste0(rootdir,"figures/Fig4_spatial_",d,"_PASTE.pdf"),width=8,height=8,pointsize = 10)
  plot(ref[,1],maxy-ref[,2],pch=16,col="grey30",lwd=0.1,xlab="",ylab="")
  points(fit[,1],maxy-fit[,2],pch=4,col='red',lwd=1,xlab="",ylab="")
  dev.off()
}
for(d in c('low','med','high')){
  ref=fig4_coords_ref
  fit=fig4_coords_gpsa[[d]]
  maxy=max(max(ref[,2]),max(fit[,2]))
  pdf(paste0(rootdir,"figures/Fig4_spatial_",d,"_GPSA.pdf"),width=8,height=8,pointsize = 10)
  plot(ref[,1],maxy-ref[,2],pch=16,col="grey30",lwd=0.1,xlab="",ylab="")
  points(fit[,1],maxy-fit[,2],pch=4,col='red',lwd=1,xlab="",ylab="")
  dev.off()
}

for(d in c('low','med','high')){
  ref=fig4_coords_ref
  fit=fig4_coords_stalign[[d]]
  maxy=max(max(ref[,2]),max(fit[,2]))
  pdf(paste0(rootdir,"figures/suppFig4_spatial_",d,"_STalign.pdf"),width=8,height=8,pointsize = 10)
  plot(ref[,1],maxy-ref[,2],pch=16,col="grey40",lwd=0.1,xlab="",ylab="")
  points(fit[,1],maxy-fit[,2],pch=4,col='red',lwd=1,xlab="",ylab="")
  dev.off()
}


df=data.frame(
  "amp"=rep(c('low','med','high'),time=6),
  "method"=c(rep("moving",3),rep('STaCker',3),rep('STUtility',3),rep('PASTE',3),rep("GPSA",3),rep("STalign",3)),
  "MSE"=c(fig4_mse_unaligned,c(mean(fig4_mse_stacker[['low']]),mean(fig4_mse_stacker[['med']]),mean(fig4_mse_stacker[['high']])),
          fig4_mse_stu,fig4_mse_paste,fig4_mse_gpsa, fig4_mse_stalign),
  'SE'=c(rep(0,3),c(sd(fig4_mse_stacker[['low']])/sqrt(length(fig4_mse_stacker[['low']])),
                    sd(fig4_mse_stacker[['med']])/sqrt(length(fig4_mse_stacker[['med']])),
                    sd(fig4_mse_stacker[['high']])/sqrt(length(fig4_mse_stacker[['high']]))),
         rep(0,3),rep(0,3), rep(0,3),rep(0,3))
)
df$method=factor(df$method,level=c('moving','STaCker','STUtility','PASTE','GPSA','STalign'))
tag='low'
#tag='med'
#tag='high'
pdf(paste0(rootdir,'figures/Fig4_MSE_',tag,'.pdf'),width=1.5,height=1.,pointsize = 20)
ggplot(df[which(tag==df$amp),]) +
  geom_bar( aes(x=method, y=MSE), stat="identity", color="black",fill="white", alpha=0.7,width=0.8) +
  geom_errorbar( aes(x=method, ymin=MSE-SE, ymax=MSE+SE), width=0.1, colour="black", alpha=0.5, size=0.5)+
  ylim(0,0.15)+coord_cartesian(ylim=c(0,0.15))+
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black"),
        axis.title.x=element_blank(),axis.text.x=element_blank(), 
        axis.title.y=element_blank())
dev.off()


(t.test(fig4_mse_stacker[['low']], mu=fig4_mse_unaligned[1]))$p.value
(t.test(fig4_mse_stacker[['med']], mu=fig4_mse_unaligned[2]))$p.value
(t.test(fig4_mse_stacker[['high']], mu=fig4_mse_unaligned[3]))$p.value
(t.test(fig4_mse_stacker[['low']], mu=fig4_mse_stu[1]))$p.value
(t.test(fig4_mse_stacker[['med']], mu=fig4_mse_stu[2]))$p.value
(t.test(fig4_mse_stacker[['high']], mu=fig4_mse_stu[3]))$p.value
(t.test(fig4_mse_stacker[['low']], mu=fig4_mse_paste[1]))$p.value
(t.test(fig4_mse_stacker[['med']], mu=fig4_mse_paste[2]))$p.value
(t.test(fig4_mse_stacker[['high']], mu=fig4_mse_paste[3]))$p.value
(t.test(fig4_mse_stacker[['low']], mu=fig4_mse_gpsa[1]))$p.value
(t.test(fig4_mse_stacker[['med']], mu=fig4_mse_gpsa[2]))$p.value
(t.test(fig4_mse_stacker[['high']], mu=fig4_mse_gpsa[3]))$p.value




########### Figure 5
load(paste0(rootdir,'data/Fig5_metrics.rda'))

slide0_normed=fig5_coords_unaligned[['0']]
slide1_normed=fig5_coords_unaligned[['1']]
slide2_normed=fig5_coords_unaligned[['2']]
slide3_normed=fig5_coords_unaligned[['3']]
maxy=max(c(max(slide0_normed[,2]),max(slide1_normed[,2]),max(slide2_normed[,2]),max(slide3_normed[,2])))
pdf(paste0(rootdir,"/figures/Fig5_spatial_unaligned.pdf"),width=8,height=8,pointsize = 11)
plot(slide0_normed[,1],maxy-slide0_normed[,2],pch=4,col="red",cex=2,lwd=2,xlab="",ylab="",xlim=c(-0.2,10.5),ylim=c(-0.2,10.5))
points(slide1_normed[,1],maxy-slide1_normed[,2],pch=4,col='green',cex=2,lwd=2,xlab="",ylab="")
points(slide2_normed[,1],maxy-slide2_normed[,2],pch=4,col='skyblue',cex=2,lwd=2,xlab="",ylab="")
points(slide3_normed[,1],maxy-slide3_normed[,2],pch=4,col='orange',cex=2,lwd=2,xlab="",ylab="")
dev.off()

slide0_normed=fig5_coords_stacker[['0']]
slide1_normed=fig5_coords_stacker[['1']]
slide2_normed=fig5_coords_stacker[['2']]
slide3_normed=fig5_coords_stacker[['3']]
maxy=max(c(max(slide0_normed[,2]),max(slide1_normed[,2]),max(slide2_normed[,2]),max(slide3_normed[,2])))
pdf(paste0(rootdir,"/figures/Fig5_spatial_stacker.pdf"),width=8,height=8,pointsize = 11)
plot(slide0_normed[,1],maxy-slide0_normed[,2],pch=4,col="red",cex=2,lwd=3,xlab="",ylab="",xlim=c(-0.2,10.5),ylim=c(-0.2,10.5))
points(slide1_normed[,1],maxy-slide1_normed[,2],pch=4,col='green',cex=2,lwd=3,xlab="",ylab="")
points(slide2_normed[,1],maxy-slide2_normed[,2],pch=4,col='skyblue',cex=2,lwd=3,xlab="",ylab="")
points(slide3_normed[,1],maxy-slide3_normed[,2],pch=4,col='orange',cex=2,lwd=3,xlab="",ylab="")
dev.off()

slide0_normed=fig5_coords_stu[['0']]
slide1_normed=fig5_coords_stu[['1']]
slide2_normed=fig5_coords_stu[['2']]
slide3_normed=fig5_coords_stu[['3']]
maxy=max(c(max(slide0_normed[,2]),max(slide1_normed[,2]),max(slide2_normed[,2]),max(slide3_normed[,2])))
pdf(paste0(rootdir,"/figures/Fig5_spatial_STUtility.pdf"),width=8,height=8,pointsize = 11)
plot(slide0_normed[,1],maxy-slide0_normed[,2],pch=4,col="red",cex=2,lwd=3,xlab="",ylab="",xlim=c(-0.2,10.5),ylim=c(-0.2,10.5))
points(slide1_normed[,1],maxy-slide1_normed[,2],pch=4,col='green',cex=2,lwd=3,xlab="",ylab="")
points(slide2_normed[,1],maxy-slide2_normed[,2],pch=4,col='skyblue',cex=2,lwd=3,xlab="",ylab="")
points(slide3_normed[,1],maxy-slide3_normed[,2],pch=4,col='orange',cex=2,lwd=3,xlab="",ylab="")
dev.off()

slide0_normed=fig5_coords_paste[['0']]
slide1_normed=fig5_coords_paste[['1']]
slide2_normed=fig5_coords_paste[['2']]
slide3_normed=fig5_coords_paste[['3']]
maxy=max(c(max(slide0_normed[,2]),max(slide1_normed[,2]),max(slide2_normed[,2]),max(slide3_normed[,2])))
pdf(paste0(rootdir,"/figures/Fig5_spatial_paste.pdf"),width=8,height=8,pointsize = 11)
plot(slide0_normed[,1],maxy-slide0_normed[,2],pch=4,col="red",cex=2,lwd=3,xlab="",ylab="",xlim=c(-0.2,10.5),ylim=c(-0.2,10.5))
points(slide1_normed[,1],maxy-slide1_normed[,2],pch=4,col='green',cex=2,lwd=3,xlab="",ylab="")
points(slide2_normed[,1],maxy-slide2_normed[,2],pch=4,col='skyblue',cex=2,lwd=3,xlab="",ylab="")
points(slide3_normed[,1],maxy-slide3_normed[,2],pch=4,col='orange',cex=2,lwd=3,xlab="",ylab="")
dev.off()

slide0_normed=fig5_coords_gpsa[['0']]
slide1_normed=fig5_coords_gpsa[['1']]
slide2_normed=fig5_coords_gpsa[['2']]
slide3_normed=fig5_coords_gpsa[['3']]
maxy=max(c(max(slide0_normed[,2]),max(slide1_normed[,2]),max(slide2_normed[,2]),max(slide3_normed[,2])))
pdf(paste0(rootdir,"/figures/Fig5_spatial_gpsa.pdf"),width=8,height=8,pointsize = 11)
plot(slide0_normed[,1],maxy-slide0_normed[,2],pch=4,col="red",cex=2,lwd=3,xlab="",ylab="",xlim=c(-0.2,10.5),ylim=c(-0.2,10.5))
points(slide1_normed[,1],maxy-slide1_normed[,2],pch=4,col='green',cex=2,lwd=3,xlab="",ylab="")
points(slide2_normed[,1],maxy-slide2_normed[,2],pch=4,col='skyblue',cex=2,lwd=3,xlab="",ylab="")
points(slide3_normed[,1],maxy-slide3_normed[,2],pch=4,col='orange',cex=2,lwd=3,xlab="",ylab="")
dev.off()

# compute standard errors
sd(fig5_mse_unaligned)/sqrt(length((fig5_mse_unaligned)))
sd(fig5_mse_stacker)/sqrt(length((fig5_mse_stacker)))
sd(fig5_mse_stu)/sqrt(length(fig5_mse_stu))
sd(fig5_mse_paste)/sqrt(length((fig5_mse_paste)))
sd(fig5_mse_gpsa)/sqrt(length((fig5_mse_gpsa)))

(t.test(fig5_mse_stacker, fig5_mse_unaligned))$p.value
(t.test(fig5_mse_stacker, fig5_mse_stu))$p.value
(t.test(fig5_mse_stacker, fig5_mse_paste))$p.value

################## Figure 6
load(paste0(rootdir,'data/Fig6_metrics.rda'))


pdf(paste0(rootdir,"/figures/Fig6_spatial_unaligned.pdf"),width=3.5,height=4.5,pointsize=10)
plot(c(-0.5,10.),c(0.6,-10.5),col='white',xlab='',ylab='')
for(c in 1:length(unique(fig6_classlabel[,'cluster']))){
  sel=rownames(fig6_classlabel)[which(fig6_classlabel[,'cluster']==unique(fig6_classlabel[,'cluster'])[c])]
  points(fig6_coords_unaligned[sel,1],-fig6_coords_unaligned[sel,2],col=colorbank[c],pch=19,cex=0.3,lwd=0.4)
}
dev.off()

pdf(paste0(rootdir,"/figures/Fig6_spatial_stacker.pdf"),width=3.5,height=4.5,pointsize=10)
plot(c(-0.5,10.),c(0.6,-10.5),col='white',xlab='',ylab='')
for(c in 1:length(unique(fig6_classlabel[,'cluster']))){
  sel=rownames(fig6_classlabel)[which(fig6_classlabel[,'cluster']==unique(fig6_classlabel[,'cluster'])[c])]
  points(fig6_coords_stacker[sel,1],-fig6_coords_stacker[sel,2],col=colorbank[c],pch=19,cex=0.3,lwd=0.4)
}
dev.off()

pdf(paste0(rootdir,"/figures/Fig6_spatial_PASTE.pdf"),width=3.5,height=4.5,pointsize=10)
plot(c(-0.5,10.),c(0.6,-10.5),col='white',xlab='',ylab='')
for(c in 1:length(unique(fig6_classlabel[,'cluster']))){
  sel=rownames(fig6_classlabel)[which(fig6_classlabel[,'cluster']==unique(fig6_classlabel[,'cluster'])[c])]
  points(fig6_coords_paste[sel,1],-fig6_coords_paste[sel,2],col=colorbank[c],pch=19,cex=0.3,lwd=0.4)
}
dev.off()

pdf(paste0(rootdir,"/figures/Fig6_spatial_GPSA.pdf"),width=3.5,height=4.5,pointsize=10)
plot(c(-0.5,10.),c(0.6,-10.5),col='white',xlab='',ylab='')
for(c in 1:length(unique(fig6_classlabel[,'cluster']))){
  sel=rownames(fig6_classlabel)[which(fig6_classlabel[,'cluster']==unique(fig6_classlabel[,'cluster'])[c])]
  points(fig6_coords_gpsa[sel,1],-fig6_coords_gpsa[sel,2],col=colorbank[c],pch=19,cex=0.3,lwd=0.4)
}
dev.off()


pdf(paste0(rootdir,"/figures/Fig6_spatial_STUtility.pdf"),width=3.5,height=4.5,pointsize=10)
plot(c(-0.5,10.),c(0.6,-10.5),col='white',xlab='',ylab='')
for(c in 1:length(unique(fig6_classlabel[,'cluster']))){
  sel=rownames(fig6_classlabel)[which(fig6_classlabel[,'cluster']==unique(fig6_classlabel[,'cluster'])[c])]
  points(fig6_coords_stu[['ref1']][sel,1],-fig6_coords_stu[['ref1']][sel,2],col=colorbank[c],pch=19,cex=0.3,lwd=0.4)
}
dev.off()

#SCS
df=data.frame(
  "method"=c(rep("unaligned",1),rep('STaCker',1),rep('STUtility',1),rep('PASTE',1)),
  "SCS"=c(mean(fig6_scs_unaligned),mean(fig6_scs_stacker),
          mean(c(mean(fig6_scs_stu1), mean(fig6_scs_stu2),mean(fig6_scs_stu3),mean(fig6_scs_stu4))),
          mean(fig6_scs_paste)),
  'SE'=c(sqrt(var(fig6_scs_unaligned))/length(fig6_scs_unaligned),
         sqrt(var(fig6_scs_stacker))/length(fig6_scs_stacker),
         sqrt(var(c(mean(fig6_scs_stu1), mean(fig6_scs_stu2),mean(fig6_scs_stu3),mean(fig6_scs_stu4))))/4,
         sqrt(var(fig6_scs_paste))/length(fig6_scs_paste) )
  
)
df$method=factor(df$method,level=c('unaligned','STaCker','STUtility','PASTE'))

pdf(paste0(rootdir,"figures/Fig6_SCS.pdf"),width=2*4/5,height=1.1,pointsize = 20)
ggplot(df[which(df$method!="GPSA"),]) +
  geom_bar( aes(x=method, y=-SCS), stat="identity", color="black",fill="white", alpha=0.7,width=0.7) +
  geom_errorbar( aes(x=method, ymin=-SCS-SE, ymax=-SCS+SE), width=0.1, colour="black", alpha=0.5, size=0.5)+
  #ylim(0,0.15)+coord_cartesian(ylim=c(0,0.15))+
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black"),
        axis.title.x=element_blank(),axis.text.x=element_blank(), 
        axis.title.y=element_blank())
dev.off()


df=data.frame('ari'=c(mean(fig6_ari_unaligned),mean(fig6_ari_stacker),mean(fig6_ari_stu),
                      mean(fig6_ari_paste)),
              'method'=c('unaligned','STaCker','STutility','PASTE'),
              'SE'=c(sqrt(var(fig6_ari_unaligned))/length(fig6_ari_unaligned),
                     sqrt(var(fig6_ari_stacker))/length(fig6_ari_stacker),
                     sqrt(var(fig6_ari_stu))/length(fig6_ari_stu),
                     sqrt(var(fig6_ari_paste))/length(fig6_ari_paste))
)
df$method=factor(df$method,level=c('unaligned','STaCker','STutility','PASTE'),order=T)

pdf(paste0(rootdir,'figures/Fig6_LARI','.pdf'),width=2*4/5,height=1.1,pointsize = 20)
ggplot(df) + 
  geom_bar( aes(x=method, y=ari), stat="identity", color="black",fill="grey", alpha=0.7,width=0.7) +
  geom_errorbar( aes(x=method, ymin=ari-SE, ymax=ari+SE), width=0.1, colour="black", alpha=0.5, size=0.5)+
  #ylim(0,0.15)+coord_cartesian(ylim=c(0,0.15))+
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black"),
        axis.title.x=element_blank(),axis.text.x=element_blank(), 
        axis.title.y=element_blank())
dev.off()

pdf(paste0(rootdir,'figures/Fig6_autocorr_unaligned.pdf'),width=10,height=6,pointsize = 2)
SpatialPlot(fig6_unaligned_seurat, features=c('AQP4',"PCP4","KRT17"),
            image.alpha=0.0,alpha=c(0.1,1),pt.size.factor = 2.6,
            ncol=3)
dev.off()

pdf(paste0(rootdir,'figures/Fig6_autocorr_stacker.pdf'),width=10,height=6,pointsize = 2)
SpatialPlot(fig6_stacker_seurat, features=c('AQP4',"PCP4","KRT17"),
            image.alpha=0.0,alpha=c(0.1,1),pt.size.factor = 2.6,
            ncol=3)
dev.off()

pdf(paste0(rootdir,'figures/Fig6_autocorr_STU.pdf'),width=10,height=6,pointsize = 2)
SpatialPlot(fig6_stu_seurat, features=c('AQP4',"PCP4","KRT17"),
            image.alpha=0.0,alpha=c(0.1,1),pt.size.factor = 2.6,
            ncol=3)
dev.off()

pdf(paste0(rootdir,'figures/Fig6_autocorr_PASTE.pdf'),width=10,height=6,pointsize = 2)
SpatialPlot(fig6_paste_seurat, features=c('AQP4',"PCP4","KRT17"),
            image.alpha=0.0,alpha=c(0.1,1),pt.size.factor = 2.6,
            ncol=3)
dev.off()


test=data.frame('gene'=rownames(fig6_moran_unaligned),
                'unaligned'=fig6_moran_unaligned,
                'STaCker'=fig6_moran_stacker,
                'STUtility'=rowMeans(cbind(fig6_moran_stu_0,fig6_moran_stu_1,fig6_moran_stu_2,fig6_moran_stu_3)),
                'PASTE'=fig6_moran_paste
                )
df=melt(test)
colnames(df)[2:3]=c('method','moranI')
df$SE=0
df$SE[which(df$method=='STUtility')]=rowSds(cbind(fig6_moran_stu_0,fig6_moran_stu_1,fig6_moran_stu_2,fig6_moran_stu_3))/sqrt(4)
df$gene=factor(df$gene,level=c('FABP7','AQP4','ENC1','HPCAL1',"CARTPT","RORB","PCP4","KRT17"),order=T)
pdf(paste0(rootdir,'figures/Fig6_moranI.pdf'),width=6,height=3.2,pointsize = 5)
ggplot(df,aes(x=gene,y=moranI,group=method,fill=method))+geom_bar(stat='identity',position='dodge')+
  geom_errorbar( aes( ymin=moranI-SE, ymax=moranI+SE),width = 0.1,position=position_dodge(width=0.9), colour="black", alpha=0.5, size=0.2)+
  theme_classic()
dev.off()


pdf(paste0(rootdir,'figures/suppFig6_DLPFC_autocorr_unaligned.pdf'),width=10,height=6,pointsize = 2)
SpatialPlot(fig6_unaligned_seurat, features=c('FABP7','ENC1','HPCAL1',"CARTPT","RORB"),
            image.alpha=0.0,alpha=c(0.1,1),pt.size.factor = 2.6,
            ncol=5)
dev.off()

pdf(paste0(rootdir,'figures/suppFig6_DLPFC_autocorr_stacker.pdf'),width=10,height=6,pointsize = 2)
SpatialPlot(fig6_stacker_seurat, features=c('FABP7','ENC1','HPCAL1',"CARTPT","RORB"),
            image.alpha=0.0,alpha=c(0.1,1),pt.size.factor = 2.6,
            ncol=5)
dev.off()

pdf(paste0(rootdir,'figures/suppFig6_DLPFC_autocorr_STU.pdf'),width=10,height=6,pointsize = 2)
SpatialPlot(fig6_stu_seurat, features=c('FABP7','ENC1','HPCAL1',"CARTPT","RORB"),
            image.alpha=0.0,alpha=c(0.1,1),pt.size.factor = 2.6,
            ncol=3)
dev.off()

pdf(paste0(rootdir,'figures/suppFig6_DLPFC_autocorr_PASTE.pdf'),width=10,height=6,pointsize = 2)
SpatialPlot(fig6_paste_seurat, features=c('FABP7','ENC1','HPCAL1',"CARTPT","RORB"),
            image.alpha=0.0,alpha=c(0.1,1),pt.size.factor = 2.6,
            ncol=3)
dev.off()


################## Figure 7
load(paste0(rootdir,'data/Fig7_metrics.rda'))

pdf(paste0(rootdir,"/figures/Fig7_spatial_unaligned.pdf"),width=4.2,height=4.2,pointsize=10)
plot(c(0,10.5),c(0,-14),col='white',xlab='',ylab='')
for(c in 1:length(unique(fig7_classlabel[,'cluster']))){
  sel=rownames(fig7_classlabel)[which(fig7_classlabel[,'cluster']==unique(fig7_classlabel[,'cluster'])[c])]
  points(fig7_coords_unaligned[sel,1],-fig7_coords_unaligned[sel,2],col=minicolorbank[c],pch=19,cex=0.8,lwd=1.5)
} #pch=16
dev.off()

pdf(paste0(rootdir,'/figures/Fig7_spatial_STaCker','.pdf'),width=4.5,height=4.,pointsize=10)
plot(c(0,10.5),c(0,-10.5),col='white',xlab='',ylab='')
for(c in 1:length(unique(fig7_classlabel[,'cluster']))){
  sel=rownames(fig7_classlabel)[which(fig7_classlabel[,'cluster']==unique(fig7_classlabel[,'cluster'])[c])]
  points(fig7_coords_stacker[sel,1],-fig7_coords_stacker[sel,2],col=minicolorbank[c],pch=19,cex=0.8,lwd=1.5)
} #pch=16
dev.off()

pdf(paste0(rootdir,'figures/Fig7_spatial_PASTE.pdf'),width=4.5,height=4.,pointsize=10)
plot(c(0,10.5),c(0,-10.5),col='white',xlab='',ylab='')
for(c in 1:length(unique(fig7_classlabel[,'cluster']))){
  sel=rownames(fig7_classlabel)[which(fig7_classlabel[,'cluster']==unique(fig7_classlabel[,'cluster'])[c])]
  points(fig7_coords_paste[sel,1],-fig7_coords_paste[sel,2],col=minicolorbank[c],pch=19,cex=0.8,lwd=1.5)
} #pch=16
dev.off()

pdf(paste0(rootdir,'figures/Fig7_spatial_GPSA.pdf'),width=4.5,height=4.5,pointsize=10)
plot(c(-1,11),c(2,-13),col='white',xlab='',ylab='')
for(c in 1:length(unique(fig7_classlabel[,'cluster']))){
  sel=rownames(fig7_classlabel)[which(fig7_classlabel[,'cluster']==unique(fig7_classlabel[,'cluster'])[c])]
  points(fig7_coords_gpsa[sel,1],-fig7_coords_gpsa[sel,2],col=minicolorbank[c],pch=19,cex=0.8,lwd=1.5)
} #pch=16
dev.off()

pdf(paste0(rootdir,'figures/Fig7_spatial_STUtility.pdf'),width=4.5,height=4.2,pointsize=10)
  plot(c(0,10.5),c(1.8,-12),col='white',xlab='',ylab='') 
  for(c in 1:length(unique(fig7_classlabel[,'cluster']))){
    sel=rownames(fig7_classlabel)[which(fig7_classlabel[,'cluster']==unique(fig7_classlabel[,'cluster'])[c])]
    points(fig7_coords_stu[['ref1']][sel,1],-fig7_coords_stu[['ref1']][sel,2],col=minicolorbank[c],pch=19,cex=0.8,lwd=1.5)
  } #pch=16
dev.off()


#####
df=data.frame(
  "method"=c(rep("unaligned",1),rep('STaCker',1),rep('STUtility',1),rep('PASTE',1)),
  "SCS"=c(mean(fig7_scs_unaligned),mean(fig7_scs_stacker),
          mean(c(mean(fig7_scs_stu1), mean(fig7_scs_stu2), mean(fig7_scs_stu3), mean(fig7_scs_stu4))),
          mean(fig7_scs_paste)
          ),
  'SE'=c(sqrt(var(fig7_scs_unaligned))/length(fig7_scs_unaligned),
         sqrt(var(fig7_scs_stacker))/length(fig7_scs_stacker),
         sqrt(var(c(mean(fig7_scs_stu1), mean(fig7_scs_stu2), mean(fig7_scs_stu3), mean(fig7_scs_stu4))))/4,
         sqrt(var(fig7_scs_paste))/length(fig7_scs_paste)
         )
)
df$method=factor(df$method,level=c('unaligned','STaCker','STUtility','PASTE','GPSA'))

pdf(paste0(rootdir,'figures/Fig7_SCS','.pdf'),width=1.8,height=1.,pointsize = 20)
ggplot(df) +
  geom_bar( aes(x=method, y=-SCS), stat="identity", color="black",fill="white", alpha=0.7,width=0.7) +
  geom_errorbar( aes(x=method, ymin=-SCS-SE, ymax=-SCS+SE), width=0.1, colour="black", alpha=0.5, size=0.5)+
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black"),
        axis.title.x=element_blank(),axis.text.x=element_blank(), 
        axis.title.y=element_blank())
dev.off()

fig7_ari_stu=c(mean(fig7_ari_stu1), mean(fig7_ari_stu2), mean(fig7_ari_stu3), mean(fig7_ari_stu4))
df=data.frame('ari'=c(mean(fig7_ari_unaligned),mean(fig7_ari_stacker), mean(fig7_ari_stu),
                      mean(fig7_ari_paste)),
              'method'=c('unaligned','STaCker','STutility','PASTE'),
              'SE'=c(sqrt(var(fig7_ari_unaligned))/length(fig7_ari_unaligned),
                     sqrt(var(fig7_ari_stacker))/length(fig7_ari_stacker),
                     sqrt(var(fig7_ari_stu))/length(fig7_ari_stu),
                     sqrt(var(fig7_ari_paste))/length(fig7_ari_paste)
                     )
)
df$method=factor(df$method,level=c('unaligned','STaCker','STutility','PASTE'),order=T)

pdf(paste0(rootdir,'figures/Fig7_LARI','.pdf'),width=1.8,height=1.,pointsize = 20)
ggplot(df[which(df$method!='GPSA'),]) +
  geom_bar( aes(x=method, y=ari), stat="identity", color="black",fill="grey", alpha=0.7,width=0.7) +
  geom_errorbar( aes(x=method, ymin=ari-SE, ymax=ari+SE), width=0.1, colour="black", alpha=0.5, size=0.5)+
  #ylim(0,0.15)+coord_cartesian(ylim=c(0,0.15))+
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black"),
        axis.title.x=element_blank(),axis.text.x=element_blank(), 
        axis.title.y=element_blank())
dev.off()

### gene pattern
pdf(paste0(rootdir,'figures/Fig7_autocorr_unaligned.pdf'),width=15,height=5,pointsize = 2)
SpatialPlot(fig7_unaligned_seurat, features=c('Spon1','Uchl1','Nrxn3'),
            image.alpha=0.0,alpha=c(0.15,1),pt.size.factor = 1,
            ncol=3)
dev.off()

pdf(paste0(rootdir,'figures/Fig7_autocorr_stacker.pdf'),width=15,height=5,pointsize = 2)
SpatialPlot(fig7_stacker_seurat, features=c('Spon1','Uchl1','Nrxn3'),
            image.alpha=0.0,alpha=c(0.15,1),pt.size.factor = 0.2,
            ncol=3)
dev.off()

pdf(paste0(rootdir,'figures/Fig7_autocorr_PASTE.pdf'),width=15,height=5,pointsize = 2)
SpatialPlot(fig7_paste_seurat, features=c('Spon1','Uchl1','Nrxn3'),
            image.alpha=0.0,alpha=c(0.1,1),pt.size.factor = 0.8,
            ncol=3)
dev.off()

pdf(paste0(rootdir,'figures/Fig7_autocorr_STUtility.pdf'),width=15,height=5,pointsize = 2)
SpatialPlot(fig7_stu_seurat, features=c('Spon1','Uchl1','Nrxn3'),
            image.alpha=0.0,alpha=c(0.1,1),pt.size.factor = 0.8,
            ncol=3)
dev.off()

pdf(paste0(rootdir,'figures/Fig7_autocorr_STalign.pdf'),width=15,height=5,pointsize = 2)
SpatialPlot(fig7_stalign_seurat, features=c('Spon1','Uchl1','Nrxn3'),
            image.alpha=0.0,alpha=c(0.1,1),pt.size.factor = 0.8,
            ncol=3)
dev.off()


#moranI
test=data.frame('gene'=rownames(fig7_moran_unaligned),
                'unaligned'=fig7_moran_unaligned[,'moran'],
                'STaCker'=fig7_moran_stacker[,'moran'],
                'STUtility'=rowMeans(cbind(fig7_moran_stu_0[,'moran'],fig7_moran_stu_1[,'moran'],fig7_moran_stu_2[,'moran'],fig7_moran_stu_3[,'moran'])),
                'PASTE'=fig7_moran_paste[,'moran']
)
df=melt(test)
colnames(df)[2:3]=c('method','moranI')
df$SE=0
df$SE[which(df$method=='STUtility')]=rowSds(cbind(fig7_moran_stu_0[,'moran'],fig7_moran_stu_1[,'moran'],fig7_moran_stu_2[,'moran'],fig7_moran_stu_3[,'moran']))/sqrt(4)
df$gene=factor(df$gene,level=c('Uchl1','Spon1','Nrxn3'),order=T)
pdf(paste0(rootdir,'figures/Fig7_moranI.pdf'),width=4.,height=2,pointsize = 2)
ggplot(df,aes(x=gene,y=moranI,group=method,fill=method))+geom_bar(stat='identity',position='dodge')+
  geom_errorbar( aes( ymin=moranI-SE, ymax=moranI+SE),width = 0.01,position=position_dodge(width=0.9), colour="black", alpha=0.5, size=0.2)+
  theme_classic()
dev.off()


############### Figure 8
load(paste0(rootdir,'data/Fig8_metrics.rda'))

# spatial plots
pdf(paste0(rootdir,'/figures/Fig8_spatial_unaligned.pdf'),width=3.5,height=4,pointsize=10)
plot(c(-0.5,10.),c(0.,-13.5),col='white',xlab='',ylab='')
for(c in 1:length(unique(fig8_classlabel[,'cluster']))){
  sel=rownames(fig8_classlabel)[which(fig8_classlabel[,'cluster']==unique(fig8_classlabel[,'cluster'])[c])]
  points(fig8_coords_unaligned[sel,1],-fig8_coords_unaligned[sel,2],col=colorbank[c],pch=1,cex=0.01,lwd=0.1)
}
dev.off()

pdf(paste0(rootdir,'/figures/Fig8_spatial_STaCker.pdf'),width=3.5,height=4,pointsize=10)
plot(c(-0.,10.),c(1.0,-12),col='white',xlab='',ylab='')
for(c in 1:length(unique(fig8_classlabel[,'cluster']))){
  sel=intersect(rownames(fig8_coords_stacker),rownames(fig8_classlabel)[which(fig8_classlabel[,'cluster']==unique(fig8_classlabel[,'cluster'])[c])])
  points(fig8_coords_stacker[sel,1],-fig8_coords_stacker[sel,2],col=colorbank[c],pch=1,cex=0.01,lwd=0.1)
}
dev.off()

pdf(paste0(rootdir,'figures/Fig8_spatial_STUtility.pdf'),width=3.5,height=4,pointsize=10)
plot(c(-0.5,10.),c(1,-11.5),col='white',xlab='',ylab='')
for(c in 1:length(unique(fig8_classlabel[,'cluster']))){
  sel=intersect(rownames(fig8_coords_stu[['ref0']]),rownames(fig8_classlabel)[which(fig8_classlabel[,'cluster']==unique(fig8_classlabel[,'cluster'])[c])])
  points(fig8_coords_stu[['ref0']][sel,1],-fig8_coords_stu[['ref0']][sel,2],col=colorbank[c],pch=1,cex=0.01,lwd=0.1)
}
dev.off()

pdf(paste0(rootdir,'figures/Fig8_spatial_stalign.pdf'),width=3.5,height=4,pointsize=10)
plot(c(-0.,10.),c(0,-10.5),col='white',xlab='',ylab='')
for(c in 1:length(unique(fig8_classlabel[,'cluster']))){
  sel=intersect(rownames(fig8_coords_stalign[['ref0']]),rownames(fig8_classlabel)[which(fig8_classlabel[,'cluster']==unique(fig8_classlabel[,'cluster'])[c])])
  points(fig8_coords_stalign[['ref0']][sel,1],-fig8_coords_stalign[['ref0']][sel,2],col=colorbank[c],pch=1,cex=0.01,lwd=0.1)
}
dev.off()

pdf(paste0(rootdir,'/figures/suppFig10_spatial_unaligned_sub0.1.pdf'),width=3.5,height=4,pointsize=10)
plot(c(-0.5,10.),c(0.,-13.5),col='white',xlab='',ylab='')
for(c in 1:length(unique(fig8_classlabel[,'cluster']))){
  sel=intersect(rownames(fig8_coords_unaligned_sub0.1),rownames(fig8_classlabel)[which(fig8_classlabel[,'cluster']==unique(fig8_classlabel[,'cluster'])[c])])
  points(fig8_coords_unaligned_sub0.1[sel,1],-fig8_coords_unaligned_sub0.1[sel,2],col=colorbank[c],pch=16,cex=0.18,lwd=0.1)
}
dev.off()

pdf(paste0(rootdir,'figures/suppFig10_spatial_stacker_sub0.1.pdf'),width=3.5,height=4,pointsize=10)
plot(c(-0.,10.),c(1.,-12.),col='white',xlab='',ylab='')
for(c in 1:length(unique(fig8_classlabel[,'cluster']))){
  sel=intersect(rownames(fig8_coords_stacker_sub0.1),rownames(fig8_classlabel)[which(fig8_classlabel[,'cluster']==unique(fig8_classlabel[,'cluster'])[c])])
  points(fig8_coords_stacker_sub0.1[sel,1],-fig8_coords_stacker_sub0.1[sel,2],col=colorbank[c],pch=16,cex=0.18,lwd=0.1)
}
dev.off()

pdf(paste0(rootdir,'figures/suppFig10_spatial_paste_sub0.1.pdf'),width=3.5,height=4,pointsize=10)
plot(c(-0.5,10.),c(-0.5,10.),col='white',xlab='',ylab='')
for(c in 1:length(unique(fig8_classlabel[,'cluster']))){
  sel=intersect(rownames(fig8_coords_paste_sub0.1),rownames(fig8_classlabel)[which(fig8_classlabel[,'cluster']==unique(fig8_classlabel[,'cluster'])[c])])
  points(fig8_coords_paste_sub0.1[sel,1],fig8_coords_paste_sub0.1[sel,2],col=colorbank[c],pch=16,cex=0.18,lwd=0.1)
}
dev.off()

pdf(paste0(rootdir,'figures/suppFig10_spatial_gpsa_sub0.1.pdf'),width=3.5,height=4,pointsize=10)
plot(c(-0.5,10.),c(0,-12.5),col='white',xlab='',ylab='')
for(c in 1:length(unique(fig8_classlabel[,'cluster']))){
  sel=intersect(rownames(fig8_coords_gpsa_sub0.1),rownames(fig8_classlabel)[which(fig8_classlabel[,'cluster']==unique(fig8_classlabel[,'cluster'])[c])])
  points(fig8_coords_gpsa_sub0.1[sel,1],-fig8_coords_gpsa_sub0.1[sel,2],col=colorbank[c],pch=16,cex=0.18,lwd=0.1)
}
dev.off()

pdf(paste0(rootdir,'figures/suppFig10_spatial_STUtility_sub0.1.pdf'),width=3.5,height=4,pointsize=10)
plot(c(-0.5,10.),c(1,-11.5),col='white',xlab='',ylab='')
for(c in 1:length(unique(fig8_classlabel[,'cluster']))){
  sel=intersect(rownames(fig8_coords_stu_sub0.1[['ref0']]),rownames(fig8_classlabel)[which(fig8_classlabel[,'cluster']==unique(fig8_classlabel[,'cluster'])[c])])
  points(fig8_coords_stu_sub0.1[['ref0']][sel,1],-fig8_coords_stu_sub0.1[['ref0']][sel,2],col=colorbank[c],pch=16,cex=0.18,lwd=0.1)
}
dev.off()

pdf(paste0(rootdir,'figures/suppFig10_spatial_stalign_sub0.1.pdf'),width=3.5,height=4,pointsize=10)
plot(c(-0.5,10.),c(0.5,-10.5),col='white',xlab='',ylab='')
for(c in 1:length(unique(fig8_classlabel[,'cluster']))){
  sel=intersect(rownames(fig8_coords_stalign_sub0.1[['ref0']]),rownames(fig8_classlabel)[which(fig8_classlabel[,'cluster']==unique(fig8_classlabel[,'cluster'])[c])])
  points(fig8_coords_stalign_sub0.1[['ref0']][sel,1],-fig8_coords_stalign_sub0.1[['ref0']][sel,2],col=colorbank[c],pch=16,cex=0.18,lwd=0.1)
}
dev.off()

### SCS
df=data.frame(
  "method"=c('unaligned','STaCker','STUtility','STalign'),
  "SCS"=c(mean(fig8_scs_unaligned),mean(fig8_scs_stacker),
          mean(colMeans(fig8_scs_stu)), 
          mean(colMeans(fig8_scs_stalign)) ),
  'SE'=c(sqrt(var(fig8_scs_unaligned))/length(fig8_scs_unaligned),sqrt(var(fig8_scs_stacker))/length(fig8_scs_stacker),
         sqrt(var(colMeans(fig8_scs_stu)))/nrow(fig8_scs_stu),
         sqrt(var(colMeans(fig8_scs_stalign)))/nrow(fig8_scs_stalign))
)
df$method=factor(df$method,level=c('unaligned','STaCker','STUtility','STalign'))

pdf(paste0(rootdir,'/figures/Fig8_SCS','.pdf'),width=1.5,height=1.,pointsize = 20)
ggplot(df) +
  geom_bar( aes(x=method, y=-SCS), stat="identity", color="black",fill="white", alpha=0.7,width=0.7) +
  geom_errorbar( aes(x=method, ymin=-SCS-SE, ymax=-SCS+SE), width=0.1, colour="black", alpha=0.5, size=0.5)+
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black"),
        axis.title.x=element_blank(),axis.text.x=element_blank(), 
        axis.title.y=element_blank())
dev.off()


df=data.frame(
  "method"=c('unaligned.sub','STaCker.sub','STUtility.sub','PASTE.sub','STalign.sub'),
  "SCS"=c(mean(fig8_scs_unaligned_sub),mean(fig8_scs_stacker_sub),
          mean(colMeans(fig8_scs_stu_sub)), 
          mean(fig8_scs_paste_sub),
          mean(colMeans(fig8_scs_stalign_sub)) ),
  'SE'=c(sqrt(var(fig8_scs_unaligned_sub))/length(fig8_scs_unaligned_sub),
         sqrt(var(fig8_scs_stacker_sub))/length(fig8_scs_stacker_sub),
         sqrt(var(colMeans(fig8_scs_stu_sub)))/nrow(fig8_scs_stu_sub),
         sqrt(var(fig8_scs_paste_sub))/length(fig8_scs_paste_sub),
         sqrt(var(colMeans(fig8_scs_stalign_sub )))/nrow(fig8_scs_stalign_sub))
)
df$method=factor(df$method,level=c('unaligned.sub','STaCker.sub','STUtility.sub','PASTE.sub','STalign.sub'))

pdf(paste0(rootdir,'/figures/suppFig10_SCS_subsampled','.pdf'),width=1.8,height=1.,pointsize = 20)
ggplot(df) +
  geom_bar( aes(x=method, y=-SCS), stat="identity", color="black",fill="white", alpha=0.7,width=0.7) +
  geom_errorbar( aes(x=method, ymin=-SCS-SE, ymax=-SCS+SE), width=0.1, colour="black", alpha=0.5, size=0.5)+
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black"),
        axis.title.x=element_blank(),axis.text.x=element_blank(), 
        axis.title.y=element_blank())
dev.off()

## lari
df=data.frame('method'=c('unaligned','STaCker','STutility','STalign'),
              'ari'=c(mean(fig8_ari_unaligned),mean(fig8_ari_stacker), 
                      mean(rowMeans(fig8_ari_stu)),
                      mean(rowMeans(fig8_ari_stalign))),
              'SE'=c(sqrt(var(fig8_ari_unaligned))/length(fig8_ari_unaligned),
                     sqrt(var(fig8_ari_stacker))/length(fig8_ari_stacker),
                     sqrt(var(rowMeans(fig8_ari_stu)))/nrow(fig8_ari_stu),
                     sqrt(var(rowMeans(fig8_ari_stalign)))/nrow(fig8_ari_stalign)
              )
)
df$method=factor(df$method,level=(c('unaligned','STaCker','STutility','STalign')),order=T)

pdf(paste0(rootdir,'figures/Fig8_LARI','.pdf'),width=1.5,height=1.,pointsize = 20)
ggplot(df) +
  geom_bar( aes(x=method, y=ari), stat="identity", color="black",fill="grey", alpha=0.7,width=0.7) +
  geom_errorbar( aes(x=forcats::fct_rev(method), ymin=ari-SE, ymax=ari+SE), width=0.1, colour="black", alpha=0.5, size=0.5)+
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black"),
        axis.title.x=element_blank(),axis.text.x=element_blank(), 
        axis.title.y=element_blank())
dev.off()

df=data.frame('method'=c('unaligned.sub','STaCker.sub','STutility.sub','PASTE.sub','STalign.sub'),
              'ari'=c(mean(fig8_ari_unaligned_sub),mean(fig8_ari_stacker_sub), 
                      mean(rowMeans(fig8_ari_stu_sub)),
                      mean(fig8_ari_paste_sub), 
                      mean(rowMeans(fig8_ari_stalign_sub))),
              'SE'=c(sqrt(var(fig8_ari_unaligned_sub))/length(fig8_ari_unaligned_sub),
                     sqrt(var(fig8_ari_stacker_sub))/length(fig8_ari_stacker_sub),
                     sqrt(var(rowMeans(fig8_ari_stu_sub)))/nrow(fig8_ari_stu_sub),
                     sqrt(var(fig8_ari_paste_sub))/length(fig8_ari_paste_sub),
                     sqrt(var(rowMeans(fig8_ari_stalign_sub)))/nrow(fig8_ari_stalign_sub)
              )
)
df$method=factor(df$method,level=(c('unaligned.sub','STaCker.sub','STutility.sub','PASTE.sub','STalign.sub')),order=T)

pdf(paste0(rootdir,'figures/suppFig10_LARI_subsampled','.pdf'),width=1.8,height=1,pointsize = 20)
ggplot(df) +
  geom_bar( aes(x=(method), y=ari), stat="identity", color="black",fill="grey", alpha=0.7,width=0.7) +
  geom_errorbar( aes(x=(method), ymin=ari-SE, ymax=ari+SE), width=0.1, colour="black", alpha=0.5, size=0.5)+
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black"),
        axis.title.x=element_blank(),axis.text.x=element_blank(), 
        axis.title.y=element_blank())#+coord_flip()
dev.off()

# gene expression 
g='Htr1a'; midpoint=1.7
#g='Drd1'; midpoint=2
#g='Efemp1'; midpoint=2
#g='Gpr101';midpoint=2
df=as.data.frame(fig8_coords_unaligned)
df$exp=fig8_exp[g,rownames(df)]
df$x=round(df$x*40,0)
df$y=round(df$y*40,0)
p1=ggplot(df,aes(x,-y))+geom_tile(aes(fill = exp))+
  scale_fill_gradient2(low = "grey92",mid="coral",high = "red4", midpoint = midpoint)+
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank()
  )
pdf(paste0(rootdir,'figures/Fig8_ImgDim_',g,'_unaligned.pdf'),width=3.8,height=2.5,pointsize = 20)
p1
dev.off()
#
df=as.data.frame(fig8_coords_stacker)
df$exp=fig8_exp[g,rownames(df)]
df$x=round(df$x*40,0)
df$y=round(df$y*40,0)
p1=ggplot(df,aes(x,-y))+geom_tile(aes(fill = exp))+
  scale_fill_gradient2(low = "grey92",mid="coral",high = "red4", midpoint = midpoint)+
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank()
  )
pdf(paste0(rootdir,'figures/Fig8_ImgDim_',g,'_STaCker.pdf'),width=3.8,height=2.5,pointsize = 20)
p1
dev.off()

df=as.data.frame(fig8_coords_stalign[['ref0']])[,1:2]
colnames(df)=c('x','y')
df$exp=fig8_exp[g,rownames(df)]
df$x=round(df$x*40,0)
df$y=round(df$y*40,0)
p1=ggplot(df,aes(x,-y))+geom_tile(aes(fill = exp))+
  scale_fill_gradient2(low = "grey92",mid="coral",high = "red4", midpoint = midpoint)+
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank()
  )
pdf(paste0(rootdir,'figures/Fig8_ImgDim_',g,'_STalign.pdf'),width=3.8,height=2.5,pointsize = 20)
p1
dev.off()

df=as.data.frame(fig8_coords_stu[['ref0']])[,1:2]
colnames(df)=c('x','y')
df$exp=fig8_exp[g,rownames(df)]
df$x=round(df$x*40,0)
df$y=round(df$y*40,0)
p1=ggplot(df,aes(x,-y))+geom_tile(aes(fill = exp))+
  scale_fill_gradient2(low = "grey92",mid="coral",high = "red4", midpoint = midpoint)+
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank()
  )
pdf(paste0(rootdir,'figures/Fig8_ImgDim_',g,'_STUtility.pdf'),width=3.8,height=2.5,pointsize = 20)
p1
dev.off()


######## MoranI
selg=c('Htr1a','Drd1','Efemp1','Gpr101')
tag='pseudo50'
df1=data.frame('MoranI'=c(fig8_moran_unaligned[[tag]][,'moran'],fig8_moran_stacker[[tag]][,'moran'],
                         rowMeans(fig8_moran_stu[[tag]],na.rm=T), rowMeans(fig8_moran_stalign[[tag]],na.rm=T),
                         fig8_moran_unaligned_sub0.1[[tag]][,'moran'],fig8_moran_stacker_sub0.1[[tag]][,'moran'],
                         rowMeans(fig8_moran_stu_sub0.1[[tag]],na.rm=T),fig8_moran_paste_sub0.1[[tag]][,'moran'],
                         rowMeans(fig8_moran_stalign_sub0.1[[tag]],na.rm=T) ),
               'dMoranI'=c(fig8_moran_unaligned[[tag]][,'moran']-fig8_moran_unaligned[[tag]][,'moran'],
                           fig8_moran_stacker[[tag]][,'moran']-fig8_moran_unaligned[[tag]][,'moran'],
                           rowMeans(fig8_moran_stu[[tag]],na.rm=T)-fig8_moran_unaligned[[tag]][,'moran'], 
                           rowMeans(fig8_moran_stalign[[tag]],na.rm=T)-fig8_moran_unaligned[[tag]][,'moran'],
                           fig8_moran_unaligned_sub0.1[[tag]][,'moran']-fig8_moran_unaligned_sub0.1[[tag]][,'moran'],
                           fig8_moran_stacker_sub0.1[[tag]][,'moran']-fig8_moran_unaligned_sub0.1[[tag]][,'moran'],
                           rowMeans(fig8_moran_stu_sub0.1[[tag]],na.rm=T)-fig8_moran_unaligned_sub0.1[[tag]][,'moran'],
                           fig8_moran_paste_sub0.1[[tag]][,'moran']-fig8_moran_unaligned_sub0.1[[tag]][,'moran'],
                           rowMeans(fig8_moran_stalign_sub0.1[[tag]],na.rm=T)-fig8_moran_unaligned_sub0.1[[tag]][,'moran'] ),
               'Gene'=c(rep(rownames(fig8_moran_unaligned[[tag]]), times=4),rep(rownames(fig8_moran_unaligned_sub0.1[[tag]]), times=5)),
              'Method'=c(rep('Unaligned',nrow(fig8_moran_unaligned[[tag]])),rep('STaCker', nrow(fig8_moran_unaligned[[tag]])),
                         rep('STUtiltiy',nrow(fig8_moran_unaligned[[tag]])),rep('STAlign', nrow(fig8_moran_unaligned[[tag]])),
                         rep('Unaligned.sub',nrow(fig8_moran_unaligned_sub0.1[[tag]])),rep('STaCker.sub', nrow(fig8_moran_unaligned_sub0.1[[tag]])),
                         rep('STUtiltiy.sub',nrow(fig8_moran_unaligned_sub0.1[[tag]])),rep('PASTE.sub', nrow(fig8_moran_unaligned_sub0.1[[tag]])),
                         rep('STAlign.sub',nrow(fig8_moran_unaligned_sub0.1[[tag]]))),
              'SE'=c(rep(0,nrow(fig8_moran_unaligned[[tag]])),rep(0, nrow(fig8_moran_unaligned[[tag]])),
                     rowSds(fig8_moran_stu[[tag]])/sqrt(3),rowSds(fig8_moran_stalign[[tag]])/sqrt(3),
                     rep(0,nrow(fig8_moran_unaligned_sub0.1[[tag]])),rep(0, nrow(fig8_moran_unaligned_sub0.1[[tag]])),
                     rowSds(fig8_moran_stu_sub0.1[[tag]])/sqrt(3),rep(0, nrow(fig8_moran_unaligned_sub0.1[[tag]])),
                     rowSds(fig8_moran_stalign_sub0.1[[tag]])/sqrt(3))
)
df1$Method=factor(df1$Method, level=c(unique(df1$Method)),order=T)
df1$Gene=factor(df1$Gene, level=c(selg,setdiff(df1$Gene, selg)),order=T)
df1$resolution='50'
#
tag='pseudo100'
df2=data.frame('MoranI'=c(fig8_moran_unaligned[[tag]][,'moran'],fig8_moran_stacker[[tag]][,'moran'],
                          rowMeans(fig8_moran_stu[[tag]],na.rm=T), rowMeans(fig8_moran_stalign[[tag]],na.rm=T),
                          fig8_moran_unaligned_sub0.1[[tag]][,'moran'],fig8_moran_stacker_sub0.1[[tag]][,'moran'],
                          rowMeans(fig8_moran_stu_sub0.1[[tag]],na.rm=T),fig8_moran_paste_sub0.1[[tag]][,'moran'],
                          rowMeans(fig8_moran_stalign_sub0.1[[tag]],na.rm=T) ),
               'dMoranI'=c(fig8_moran_unaligned[[tag]][,'moran']-fig8_moran_unaligned[[tag]][,'moran'],
                           fig8_moran_stacker[[tag]][,'moran']-fig8_moran_unaligned[[tag]][,'moran'],
                           rowMeans(fig8_moran_stu[[tag]],na.rm=T)-fig8_moran_unaligned[[tag]][,'moran'], 
                           rowMeans(fig8_moran_stalign[[tag]],na.rm=T)-fig8_moran_unaligned[[tag]][,'moran'],
                           fig8_moran_unaligned_sub0.1[[tag]][,'moran']-fig8_moran_unaligned_sub0.1[[tag]][,'moran'],
                           fig8_moran_stacker_sub0.1[[tag]][,'moran']-fig8_moran_unaligned_sub0.1[[tag]][,'moran'],
                           rowMeans(fig8_moran_stu_sub0.1[[tag]],na.rm=T)-fig8_moran_unaligned_sub0.1[[tag]][,'moran'],
                           fig8_moran_paste_sub0.1[[tag]][,'moran']-fig8_moran_unaligned_sub0.1[[tag]][,'moran'],
                           rowMeans(fig8_moran_stalign_sub0.1[[tag]],na.rm=T)-fig8_moran_unaligned_sub0.1[[tag]][,'moran'] ),
               'Gene'=c(rep(rownames(fig8_moran_unaligned[[tag]]), times=4),rep(rownames(fig8_moran_unaligned_sub0.1[[tag]]), times=5)),
               'Method'=c(rep('Unaligned',nrow(fig8_moran_unaligned[[tag]])),rep('STaCker', nrow(fig8_moran_unaligned[[tag]])),
                          rep('STUtiltiy',nrow(fig8_moran_unaligned[[tag]])),rep('STAlign', nrow(fig8_moran_unaligned[[tag]])),
                          rep('Unaligned.sub',nrow(fig8_moran_unaligned_sub0.1[[tag]])),rep('STaCker.sub', nrow(fig8_moran_unaligned_sub0.1[[tag]])),
                          rep('STUtiltiy.sub',nrow(fig8_moran_unaligned_sub0.1[[tag]])),rep('PASTE.sub', nrow(fig8_moran_unaligned_sub0.1[[tag]])),
                          rep('STAlign.sub',nrow(fig8_moran_unaligned_sub0.1[[tag]]))),
               'SE'=c(rep(0,nrow(fig8_moran_unaligned[[tag]])),rep(0, nrow(fig8_moran_unaligned[[tag]])),
                      rowSds(fig8_moran_stu[[tag]])/sqrt(3),rowSds(fig8_moran_stalign[[tag]])/sqrt(3),
                      rep(0,nrow(fig8_moran_unaligned_sub0.1[[tag]])),rep(0, nrow(fig8_moran_unaligned_sub0.1[[tag]])),
                      rowSds(fig8_moran_stu_sub0.1[[tag]])/sqrt(3),rep(0, nrow(fig8_moran_unaligned_sub0.1[[tag]])),
                      rowSds(fig8_moran_stalign_sub0.1[[tag]])/sqrt(3))
)
df2$Method=factor(df1$Method, level=c(unique(df1$Method)),order=T)
df2$Gene=factor(df2$Gene, level=c(selg,setdiff(df2$Gene, selg)),order=T)
df2$resolution='100'
#
tag='pseudo200'
df3=data.frame('MoranI'=c(fig8_moran_unaligned[[tag]][,'moran'],fig8_moran_stacker[[tag]][,'moran'],
                          rowMeans(fig8_moran_stu[[tag]],na.rm=T), rowMeans(fig8_moran_stalign[[tag]],na.rm=T),
                          fig8_moran_unaligned_sub0.1[[tag]][,'moran'],fig8_moran_stacker_sub0.1[[tag]][,'moran'],
                          rowMeans(fig8_moran_stu_sub0.1[[tag]],na.rm=T),fig8_moran_paste_sub0.1[[tag]][,'moran'],
                          rowMeans(fig8_moran_stalign_sub0.1[[tag]],na.rm=T) ),
               'dMoranI'=c(fig8_moran_unaligned[[tag]][,'moran']-fig8_moran_unaligned[[tag]][,'moran'],
                           fig8_moran_stacker[[tag]][,'moran']-fig8_moran_unaligned[[tag]][,'moran'],
                           rowMeans(fig8_moran_stu[[tag]],na.rm=T)-fig8_moran_unaligned[[tag]][,'moran'], 
                           rowMeans(fig8_moran_stalign[[tag]],na.rm=T)-fig8_moran_unaligned[[tag]][,'moran'],
                           fig8_moran_unaligned_sub0.1[[tag]][,'moran']-fig8_moran_unaligned_sub0.1[[tag]][,'moran'],
                           fig8_moran_stacker_sub0.1[[tag]][,'moran']-fig8_moran_unaligned_sub0.1[[tag]][,'moran'],
                           rowMeans(fig8_moran_stu_sub0.1[[tag]],na.rm=T)-fig8_moran_unaligned_sub0.1[[tag]][,'moran'],
                           fig8_moran_paste_sub0.1[[tag]][,'moran']-fig8_moran_unaligned_sub0.1[[tag]][,'moran'],
                           rowMeans(fig8_moran_stalign_sub0.1[[tag]],na.rm=T)-fig8_moran_unaligned_sub0.1[[tag]][,'moran'] ),
               'Gene'=c(rep(rownames(fig8_moran_unaligned[[tag]]), times=4),rep(rownames(fig8_moran_unaligned_sub0.1[[tag]]), times=5)),
               'Method'=c(rep('Unaligned',nrow(fig8_moran_unaligned[[tag]])),rep('STaCker', nrow(fig8_moran_unaligned[[tag]])),
                          rep('STUtiltiy',nrow(fig8_moran_unaligned[[tag]])),rep('STAlign', nrow(fig8_moran_unaligned[[tag]])),
                          rep('Unaligned.sub',nrow(fig8_moran_unaligned_sub0.1[[tag]])),rep('STaCker.sub', nrow(fig8_moran_unaligned_sub0.1[[tag]])),
                          rep('STUtiltiy.sub',nrow(fig8_moran_unaligned_sub0.1[[tag]])),rep('PASTE.sub', nrow(fig8_moran_unaligned_sub0.1[[tag]])),
                          rep('STAlign.sub',nrow(fig8_moran_unaligned_sub0.1[[tag]]))),
               'SE'=c(rep(0,nrow(fig8_moran_unaligned[[tag]])),rep(0, nrow(fig8_moran_unaligned[[tag]])),
                      rowSds(fig8_moran_stu[[tag]])/sqrt(3),rowSds(fig8_moran_stalign[[tag]])/sqrt(3),
                      rep(0,nrow(fig8_moran_unaligned_sub0.1[[tag]])),rep(0, nrow(fig8_moran_unaligned_sub0.1[[tag]])),
                      rowSds(fig8_moran_stu_sub0.1[[tag]])/sqrt(3),rep(0, nrow(fig8_moran_unaligned_sub0.1[[tag]])),
                      rowSds(fig8_moran_stalign_sub0.1[[tag]])/sqrt(3))
)
df3$Method=factor(df3$Method, level=c(unique(df3$Method)),order=T)
df3$Gene=factor(df3$Gene, level=c(selg,setdiff(df3$Gene, selg)),order=T)
df3$resolution='200'

tag='pseudo500'
df4=data.frame('MoranI'=c(fig8_moran_unaligned[[tag]][,'moran'],fig8_moran_stacker[[tag]][,'moran'],
                          rowMeans(fig8_moran_stu[[tag]],na.rm=T), rowMeans(fig8_moran_stalign[[tag]],na.rm=T),
                          fig8_moran_unaligned_sub0.1[[tag]][,'moran'],fig8_moran_stacker_sub0.1[[tag]][,'moran'],
                          rowMeans(fig8_moran_stu_sub0.1[[tag]],na.rm=T),fig8_moran_paste_sub0.1[[tag]][,'moran'],
                          rowMeans(fig8_moran_stalign_sub0.1[[tag]],na.rm=T) ),
               'dMoranI'=c(fig8_moran_unaligned[[tag]][,'moran']-fig8_moran_unaligned[[tag]][,'moran'],
                           fig8_moran_stacker[[tag]][,'moran']-fig8_moran_unaligned[[tag]][,'moran'],
                           rowMeans(fig8_moran_stu[[tag]],na.rm=T)-fig8_moran_unaligned[[tag]][,'moran'], 
                           rowMeans(fig8_moran_stalign[[tag]],na.rm=T)-fig8_moran_unaligned[[tag]][,'moran'],
                           fig8_moran_unaligned_sub0.1[[tag]][,'moran']-fig8_moran_unaligned_sub0.1[[tag]][,'moran'],
                           fig8_moran_stacker_sub0.1[[tag]][,'moran']-fig8_moran_unaligned_sub0.1[[tag]][,'moran'],
                           rowMeans(fig8_moran_stu_sub0.1[[tag]],na.rm=T)-fig8_moran_unaligned_sub0.1[[tag]][,'moran'],
                           fig8_moran_paste_sub0.1[[tag]][,'moran']-fig8_moran_unaligned_sub0.1[[tag]][,'moran'],
                           rowMeans(fig8_moran_stalign_sub0.1[[tag]],na.rm=T)-fig8_moran_unaligned_sub0.1[[tag]][,'moran'] ),
               'Gene'=c(rep(rownames(fig8_moran_unaligned[[tag]]), times=4),rep(rownames(fig8_moran_unaligned_sub0.1[[tag]]), times=5)),
               'Method'=c(rep('Unaligned',nrow(fig8_moran_unaligned[[tag]])),rep('STaCker', nrow(fig8_moran_unaligned[[tag]])),
                          rep('STUtiltiy',nrow(fig8_moran_unaligned[[tag]])),rep('STAlign', nrow(fig8_moran_unaligned[[tag]])),
                          rep('Unaligned.sub',nrow(fig8_moran_unaligned_sub0.1[[tag]])),rep('STaCker.sub', nrow(fig8_moran_unaligned_sub0.1[[tag]])),
                          rep('STUtiltiy.sub',nrow(fig8_moran_unaligned_sub0.1[[tag]])),rep('PASTE.sub', nrow(fig8_moran_unaligned_sub0.1[[tag]])),
                          rep('STAlign.sub',nrow(fig8_moran_unaligned_sub0.1[[tag]]))),
               'SE'=c(rep(0,nrow(fig8_moran_unaligned[[tag]])),rep(0, nrow(fig8_moran_unaligned[[tag]])),
                      rowSds(fig8_moran_stu[[tag]])/sqrt(3),rowSds(fig8_moran_stalign[[tag]])/sqrt(3),
                      rep(0,nrow(fig8_moran_unaligned_sub0.1[[tag]])),rep(0, nrow(fig8_moran_unaligned_sub0.1[[tag]])),
                      rowSds(fig8_moran_stu_sub0.1[[tag]])/sqrt(3),rep(0, nrow(fig8_moran_unaligned_sub0.1[[tag]])),
                      rowSds(fig8_moran_stalign_sub0.1[[tag]])/sqrt(3))
)
df4$Method=factor(df4$Method, level=c(unique(df4$Method)),order=T)
df4$Gene=factor(df4$Gene, level=c(selg,setdiff(df4$Gene, selg)),order=T)
df4$resolution='500'

df=rbind(df1,df2,df3,df4)
df$resolution=factor(df$resolution,level=c(50,100,200,500),order=T)


###
pcut=0.01
nontrivial=nontrivial_sub0.1=list()
for(tag in names(fig8_moran_unaligned_slice0)){
  nontrivial[[tag]]=intersect(intersect(rownames(fig8_moran_unaligned_slice0[[tag]])[which(fig8_moran_unaligned_slice0[[tag]][,'pvalue']<=pcut)],
                                        rownames(fig8_moran_unaligned_slice1[[tag]])[which(fig8_moran_unaligned_slice1[[tag]][,'pvalue']<=pcut)]),
                              rownames(fig8_moran_unaligned_slice2[[tag]])[which(fig8_moran_unaligned_slice2[[tag]][,'pvalue']<=pcut)])
  nontrivial_sub0.1[[tag]]=intersect(intersect(rownames(fig8_moran_unaligned_sub0.1_slice0[[tag]])[which(fig8_moran_unaligned_sub0.1_slice0[[tag]][,'pvalue']<=pcut)],
                                               rownames(fig8_moran_unaligned_sub0.1_slice1[[tag]])[which(fig8_moran_unaligned_sub0.1_slice1[[tag]][,'pvalue']<=pcut)]),
                                     rownames(fig8_moran_unaligned_sub0.1_slice2[[tag]])[which(fig8_moran_unaligned_sub0.1_slice2[[tag]][,'pvalue']<=pcut)])
}
###
tag='pseudo50'
pdf(paste0(rootdir,'figures/Fig8_selGene_MoranI_',tag,'.pdf'),width=2.5,height=1.8,pointsize = 10)
ggplot(df[which(df$Gene %in% selg & df$Method %in% c("Unaligned",'STaCker','STUtiltiy','STAlign') & df$resolution==gsub("pseudo","",tag)),],aes(x=Gene, y=MoranI, group=Method, fill=Method))+
  geom_bar(stat="identity",position='dodge',show.legend = F)+
  geom_errorbar(aes(ymin=MoranI-SE, ymax=MoranI+SE), position =  position_dodge(width = 0.9), width=0.05, colour="black", alpha=1, size=0.1)+
  scale_colour_manual(values=c('grey','red','green2','blue','black','orange','aquamarine2','violet','cyan','gold'))+
  scale_fill_manual(values=c('grey','red','green2','blue','black','orange','aquamarine2','violet','cyan','gold'))+
  ylim(0,0.65)+
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black"),
        axis.title.x=element_blank(),#axis.text.y=element_blank(), 
        axis.title.y=element_blank(),axis.text.x = element_text(angle = 0, vjust = 1, hjust=0.5))
dev.off()

pdf(paste0(rootdir,'figures/suppFig10_selGene_MoranI_subsampled_',tag,'.pdf'),width=2.5,height=1.8,pointsize = 10)
ggplot(df[which(df$Gene %in% selg & df$Method %in% c("Unaligned.sub",'STaCker.sub','STUtiltiy.sub','PASTE.sub','STAlign.sub') & df$resolution==gsub("pseudo","",tag)),],aes(x=Gene, y=MoranI, group=Method, fill=Method))+
  geom_bar(stat="identity",position='dodge',show.legend = F)+
  geom_errorbar(aes(ymin=MoranI-SE, ymax=MoranI+SE), position =  position_dodge(width = 0.9), width=0.05, colour="black", alpha=1, size=0.1)+
  scale_colour_manual(values=c('grey','red','green2','blue','black','orange','aquamarine2','violet','cyan','gold'))+
  scale_fill_manual(values=c('grey','red','green2','blue','black','orange','aquamarine2','violet','cyan','gold'))+
  ylim(0,0.35)+
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black"),
        axis.title.x=element_blank(),#axis.text.y=element_blank(), 
        axis.title.y=element_blank(),axis.text.x = element_text(angle = 0, vjust = 1, hjust=0.5))
dev.off()

pdf(paste0(rootdir,'figures/Fig8_nonTrivialGenes_MoranI_',tag,'.pdf'),width=2.,height=1.8,pointsize = 20)
ggplot(df[which(df$Method %in% c("Unaligned",'STaCker','STUtiltiy','STAlign') & df$Gene %in% nontrivial[[tag]] &df$resolution==gsub('pseudo','',tag)),],aes(x=Method, y=MoranI, group=Method,color=Method))+
  geom_boxplot(show.legend = FALSE,width=0.5)+
  scale_colour_manual(values=c('grey','red','green2','blue','black','orange','aquamarine2','violet','cyan','gold'))+
  scale_fill_manual(values=c('grey','red','green2','blue','black','orange','aquamarine2','violet','cyan','gold'))+
  ylim(0.,0.8)+coord_cartesian(ylim=c(0.05,0.58))+
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black"),
        axis.title.x=element_blank(),#axis.text.y=element_blank(), 
        axis.title.y=element_blank(), axis.text.x=element_blank() #axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)
  )
dev.off()

pdf(paste0(rootdir,'figures/suppFig10_nonTrivialGenes_MoranI_subsampled_',tag,'.pdf'),width=2.,height=1.8,pointsize = 20)
ggplot(df[which(df$Method %in% c("Unaligned.sub",'STaCker.sub','STUtiltiy.sub','PASTE.sub','STAlign.sub')& df$Gene %in% nontrivial_sub0.1[[tag]] &df$resolution==gsub('pseudo','',tag)),],aes(x=Method, y=MoranI, group=Method,color=Method))+geom_boxplot(show.legend = FALSE)+
  scale_colour_manual(values=c('grey','red','green2','blue','black','orange','aquamarine2','violet','cyan','gold'))+
  scale_fill_manual(values=c('grey','red','green2','blue','black','orange','aquamarine2','violet','cyan','gold'))+
  ylim(0.0,0.65)+coord_cartesian(ylim=c(0.05,0.43))+
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black"),
        axis.title.x=element_blank(),#axis.text.y=element_blank(), 
        axis.title.y=element_blank(), axis.text.x=element_blank() #axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)
  )
dev.off()

tag='pseudo100'
#tag='pseudo200'
#tag='pseudo500'
pdf(paste0(rootdir,'figures/suppFig9_selGene_MoranI_',tag,'.pdf'),width=2.5,height=1.8,pointsize = 10)
ggplot(df[which(df$Gene %in% selg & df$Method %in% c("Unaligned",'STaCker','STUtiltiy','STAlign') & df$resolution==gsub("pseudo","",tag)),],aes(x=Gene, y=MoranI, group=Method, fill=Method))+
  geom_bar(stat="identity",position='dodge',show.legend = F)+
  geom_errorbar(aes(ymin=MoranI-SE, ymax=MoranI+SE), position =  position_dodge(width = 0.9), width=0.05, colour="black", alpha=1, size=0.1)+
  scale_colour_manual(values=c('grey','red','green2','blue','black','orange','aquamarine2','violet','cyan','gold'))+
  scale_fill_manual(values=c('grey','red','green2','blue','black','orange','aquamarine2','violet','cyan','gold'))+
  ylim(0,0.8)+
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black"),
        axis.title.x=element_blank(),#axis.text.y=element_blank(), 
        axis.title.y=element_blank(),axis.text.x = element_text(angle = 0, vjust = 1, hjust=0.5))
dev.off()

pdf(paste0(rootdir,'figures/suppFig11_selGene_MoranI_subsampled_',tag,'.pdf'),width=2.5,height=1.8,pointsize = 10)
ggplot(df[which(df$Gene %in% selg & df$Method %in% c("Unaligned.sub",'STaCker.sub','STUtiltiy.sub','PASTE.sub','STAlign.sub') & df$resolution==gsub("pseudo","",tag)),],aes(x=Gene, y=MoranI, group=Method, fill=Method))+
  geom_bar(stat="identity",position='dodge',show.legend = F)+
  geom_errorbar(aes(ymin=MoranI-SE, ymax=MoranI+SE), position =  position_dodge(width = 0.9), width=0.05, colour="black", alpha=1, size=0.1)+
  scale_colour_manual(values=c('grey','red','green2','blue','black','orange','aquamarine2','violet','cyan','gold'))+
  scale_fill_manual(values=c('grey','red','green2','blue','black','orange','aquamarine2','violet','cyan','gold'))+
  ylim(0,0.8)+
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black"),
        axis.title.x=element_blank(),#axis.text.y=element_blank(), 
        axis.title.y=element_blank(),axis.text.x = element_text(angle = 0, vjust = 1, hjust=0.5))
dev.off()

pdf(paste0(rootdir,'figures/suppFig9_nonTrivialGenes_MoranI_',tag,'.pdf'),width=2.,height=1.8,pointsize = 20)
ggplot(df[which(df$Method %in% c("Unaligned",'STaCker','STUtiltiy','STAlign') & df$Gene %in% nontrivial[[tag]] &df$resolution==gsub('pseudo','',tag)),],aes(x=Method, y=MoranI, group=Method,color=Method))+geom_boxplot(show.legend = FALSE)+
  scale_colour_manual(values=c('grey','red','green2','blue','black','orange','aquamarine2','violet','cyan','gold'))+
  scale_fill_manual(values=c('grey','red','green2','blue','black','orange','aquamarine2','violet','cyan','gold'))+
  ylim(0.,0.8)+coord_cartesian(ylim=c(0.07,0.7))+
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black"),
        axis.title.x=element_blank(),#axis.text.y=element_blank(), 
        axis.title.y=element_blank(), axis.text.x=element_blank() #axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)
  )
dev.off()

pdf(paste0(rootdir,'figures/suppFig11_nonTrivialGenes_MoranI_subsampled_',tag,'.pdf'),width=2.,height=1.8,pointsize = 20)
ggplot(df[which(df$Method %in% c("Unaligned.sub",'STaCker.sub','STUtiltiy.sub','PASTE.sub','STAlign.sub')& df$Gene %in% nontrivial_sub0.1[[tag]] &df$resolution==gsub('pseudo','',tag)),],aes(x=Method, y=MoranI, group=Method,color=Method))+geom_boxplot(show.legend = FALSE)+
  scale_colour_manual(values=c('grey','red','green2','blue','black','orange','aquamarine2','violet','cyan','gold'))+
  scale_fill_manual(values=c('grey','red','green2','blue','black','orange','aquamarine2','violet','cyan','gold'))+
  ylim(0.0,0.7)+
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black"),
        axis.title.x=element_blank(),#axis.text.y=element_blank(), 
        axis.title.y=element_blank(), axis.text.x=element_blank() #axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)
  )
dev.off()


###Similarity
fig8_sim_stu=fig8_sim_stalign=fig8_sim_stu_sub0.1=fig8_sim_stalign_sub0.1=list()
for(tag in c('pseudo50','pseudo100','pseudo200','pseudo500')){
fig8_sim_stu[[tag]]=cbind(rowMeans(cbind(fig8_sim_stu_0[[tag]][,"0:1"],fig8_sim_stu_1[[tag]][,"0:1"],fig8_sim_stu_2[[tag]][,"0:1"])),
                   rowMeans(cbind(fig8_sim_stu_0[[tag]][,"0:2"],fig8_sim_stu_1[[tag]][,"0:2"],fig8_sim_stu_2[[tag]][,"0:2"])),
                   rowMeans(cbind(fig8_sim_stu_0[[tag]][,"1:2"],fig8_sim_stu_1[[tag]][,"1:2"],fig8_sim_stu_2[[tag]][,"1:2"])) )
colnames(fig8_sim_stu[[tag]])=c('avg.0:1','avg.0:2','avg.1:2')
fig8_sim_stalign[[tag]]=cbind(rowMeans(cbind(fig8_sim_stalign_0[[tag]][,"0:1"],fig8_sim_stalign_1[[tag]][,"0:1"],fig8_sim_stalign_2[[tag]][,"0:1"])),
                   rowMeans(cbind(fig8_sim_stalign_0[[tag]][,"0:2"],fig8_sim_stalign_1[[tag]][,"0:2"],fig8_sim_stalign_2[[tag]][,"0:2"])),
                   rowMeans(cbind(fig8_sim_stalign_0[[tag]][,"1:2"],fig8_sim_stalign_1[[tag]][,"1:2"],fig8_sim_stalign_2[[tag]][,"1:2"])) )
colnames(fig8_sim_stalign[[tag]])=c('avg.0:1','avg.0:2','avg.1:2')

fig8_sim_stu_sub0.1[[tag]]=cbind(rowMeans(cbind(fig8_sim_stu_sub0.1_0[[tag]][,"0:1"],fig8_sim_stu_sub0.1_1[[tag]][,"0:1"],fig8_sim_stu_sub0.1_2[[tag]][,"0:1"])),
                   rowMeans(cbind(fig8_sim_stu_sub0.1_0[[tag]][,"0:2"],fig8_sim_stu_sub0.1_1[[tag]][,"0:2"],fig8_sim_stu_sub0.1_2[[tag]][,"0:2"])),
                   rowMeans(cbind(fig8_sim_stu_sub0.1_0[[tag]][,"1:2"],fig8_sim_stu_sub0.1_1[[tag]][,"1:2"],fig8_sim_stu_sub0.1_2[[tag]][,"1:2"])) )
colnames(fig8_sim_stu_sub0.1[[tag]])=c('avg.0:1','avg.0:2','avg.1:2')
fig8_sim_stalign_sub0.1[[tag]]=cbind(rowMeans(cbind(fig8_sim_stalign_sub0.1_0[[tag]][,"0:1"],fig8_sim_stalign_sub0.1_1[[tag]][,"0:1"],fig8_sim_stalign_sub0.1_2[[tag]][,"0:1"])),
                          rowMeans(cbind(fig8_sim_stalign_sub0.1_0[[tag]][,"0:2"],fig8_sim_stalign_sub0.1_1[[tag]][,"0:2"],fig8_sim_stalign_sub0.1_2[[tag]][,"0:2"])),
                          rowMeans(cbind(fig8_sim_stalign_sub0.1_0[[tag]][,"1:2"],fig8_sim_stalign_sub0.1_1[[tag]][,"1:2"],fig8_sim_stalign_sub0.1_2[[tag]][,"1:2"])) )
colnames(fig8_sim_stalign_sub0.1[[tag]])=c('avg.0:1','avg.0:2','avg.1:2')
}
##
tag='pseudo50'
df1=data.frame('Similarity'=c(rowMeans(fig8_sim_unaligned[[tag]],na.rm=T),
                             rowMeans(fig8_sim_stacker[[tag]],na.rm=T),
                             rowMeans(fig8_sim_stu[[tag]],na.rm=T),
                             rowMeans(fig8_sim_stalign[[tag]],na.rm=T),
                             rowMeans(fig8_sim_unaligned_sub0.1[[tag]],na.rm=T),
                             rowMeans(fig8_sim_stacker_sub0.1[[tag]],na.rm=T),
                             rowMeans(fig8_sim_stu_sub0.1[[tag]],na.rm=T),
                             rowMeans(fig8_sim_paste_sub0.1[[tag]],na.rm=T),
                             rowMeans(fig8_sim_stalign_sub0.1[[tag]],na.rm=T)),
              'Gene'=c(rownames(fig8_sim_unaligned[[tag]]), rownames(fig8_sim_stacker[[tag]]),
                       rownames(fig8_sim_stu[[tag]]), rownames(fig8_sim_stalign[[tag]]),
                       rownames(fig8_sim_unaligned_sub0.1[[tag]]), rownames(fig8_sim_stacker_sub0.1[[tag]]),
                       rownames(fig8_sim_stu_sub0.1[[tag]]), rownames(fig8_sim_paste_sub0.1[[tag]]),
                       rownames(fig8_sim_stalign_sub0.1[[tag]])),
              'SE'=c(rowSds(fig8_sim_unaligned[[tag]])/sqrt(3),rowSds(fig8_sim_stacker[[tag]])/sqrt(3),
                     rowSds(fig8_sim_stu[[tag]])/sqrt(3),
                     rowSds(fig8_sim_stalign[[tag]])/sqrt(3),
                     rowSds(fig8_sim_unaligned_sub0.1[[tag]])/sqrt(3),rowSds(fig8_sim_stacker_sub0.1[[tag]])/sqrt(3),
                     rowSds(fig8_sim_stu_sub0.1[[tag]])/sqrt(3),
                     rowSds(fig8_sim_paste_sub0.1[[tag]])/sqrt(3),
                     rowSds(fig8_sim_stalign_sub0.1[[tag]])/sqrt(3)),
              'Method'=c(rep('Unaligned',nrow(fig8_sim_unaligned[[tag]])),rep('STaCker', nrow(fig8_sim_stacker[[tag]])),
                         rep('STUtiltiy',nrow(fig8_sim_stu[[tag]])),rep('STAlign', nrow(fig8_sim_stalign[[tag]])),
                         rep('Unaligned.sub',nrow(fig8_sim_unaligned_sub0.1[[tag]])),rep('STaCker.sub', nrow(fig8_sim_stacker_sub0.1[[tag]])),
                         rep('STUtiltiy.sub',nrow(fig8_sim_stu_sub0.1[[tag]])),rep('PASTE.sub', nrow(fig8_sim_paste_sub0.1[[tag]])),
                         rep('STAlign.sub',nrow(fig8_sim_stalign_sub0.1[[tag]])))
)
df1$Method=factor(df1$Method, level=c(unique(df1$Method)),order=T)
df1$Gene=factor(df1$Gene, level=c(selg,setdiff(df1$Gene, selg)),order=T)
df1$resolution='50'

tag='pseudo100'
df2=data.frame('Similarity'=c(rowMeans(fig8_sim_unaligned[[tag]],na.rm=T),
                              rowMeans(fig8_sim_stacker[[tag]],na.rm=T),
                              rowMeans(fig8_sim_stu[[tag]],na.rm=T),
                              rowMeans(fig8_sim_stalign[[tag]],na.rm=T),
                              rowMeans(fig8_sim_unaligned_sub0.1[[tag]],na.rm=T),
                              rowMeans(fig8_sim_stacker_sub0.1[[tag]],na.rm=T),
                              rowMeans(fig8_sim_stu_sub0.1[[tag]],na.rm=T),
                              rowMeans(fig8_sim_paste_sub0.1[[tag]],na.rm=T),
                              rowMeans(fig8_sim_stalign_sub0.1[[tag]],na.rm=T)),
               'Gene'=c(rownames(fig8_sim_unaligned[[tag]]), rownames(fig8_sim_stacker[[tag]]),
                        rownames(fig8_sim_stu[[tag]]), rownames(fig8_sim_stalign[[tag]]),
                        rownames(fig8_sim_unaligned_sub0.1[[tag]]), rownames(fig8_sim_stacker_sub0.1[[tag]]),
                        rownames(fig8_sim_stu_sub0.1[[tag]]), rownames(fig8_sim_paste_sub0.1[[tag]]),
                        rownames(fig8_sim_stalign_sub0.1[[tag]])),
               'SE'=c(rowSds(fig8_sim_unaligned[[tag]])/sqrt(3),rowSds(fig8_sim_stacker[[tag]])/sqrt(3),
                      rowSds(fig8_sim_stu[[tag]])/sqrt(3),
                      rowSds(fig8_sim_stalign[[tag]])/sqrt(3),
                      rowSds(fig8_sim_unaligned_sub0.1[[tag]])/sqrt(3),rowSds(fig8_sim_stacker_sub0.1[[tag]])/sqrt(3),
                      rowSds(fig8_sim_stu_sub0.1[[tag]])/sqrt(3),
                      rowSds(fig8_sim_paste_sub0.1[[tag]])/sqrt(3),
                      rowSds(fig8_sim_stalign_sub0.1[[tag]])/sqrt(3)),
               'Method'=c(rep('Unaligned',nrow(fig8_sim_unaligned[[tag]])),rep('STaCker', nrow(fig8_sim_stacker[[tag]])),
                          rep('STUtiltiy',nrow(fig8_sim_stu[[tag]])),rep('STAlign', nrow(fig8_sim_stalign[[tag]])),
                          rep('Unaligned.sub',nrow(fig8_sim_unaligned_sub0.1[[tag]])),rep('STaCker.sub', nrow(fig8_sim_stacker_sub0.1[[tag]])),
                          rep('STUtiltiy.sub',nrow(fig8_sim_stu_sub0.1[[tag]])),rep('PASTE.sub', nrow(fig8_sim_paste_sub0.1[[tag]])),
                          rep('STAlign.sub',nrow(fig8_sim_stalign_sub0.1[[tag]])))
)
df2$Method=factor(df2$Method, level=c(unique(df2$Method)),order=T)
df2$Gene=factor(df2$Gene, level=c(selg,setdiff(df2$Gene, selg)),order=T)
df2$resolution='100'

tag='pseudo200'
df3=data.frame('Similarity'=c(rowMeans(fig8_sim_unaligned[[tag]],na.rm=T),
                              rowMeans(fig8_sim_stacker[[tag]],na.rm=T),
                              rowMeans(fig8_sim_stu[[tag]],na.rm=T),
                              rowMeans(fig8_sim_stalign[[tag]],na.rm=T),
                              rowMeans(fig8_sim_unaligned_sub0.1[[tag]],na.rm=T),
                              rowMeans(fig8_sim_stacker_sub0.1[[tag]],na.rm=T),
                              rowMeans(fig8_sim_stu_sub0.1[[tag]],na.rm=T),
                              rowMeans(fig8_sim_paste_sub0.1[[tag]],na.rm=T),
                              rowMeans(fig8_sim_stalign_sub0.1[[tag]],na.rm=T)),
               'Gene'=c(rownames(fig8_sim_unaligned[[tag]]), rownames(fig8_sim_stacker[[tag]]),
                        rownames(fig8_sim_stu[[tag]]), rownames(fig8_sim_stalign[[tag]]),
                        rownames(fig8_sim_unaligned_sub0.1[[tag]]), rownames(fig8_sim_stacker_sub0.1[[tag]]),
                        rownames(fig8_sim_stu_sub0.1[[tag]]), rownames(fig8_sim_paste_sub0.1[[tag]]),
                        rownames(fig8_sim_stalign_sub0.1[[tag]])),
               'SE'=c(rowSds(fig8_sim_unaligned[[tag]])/sqrt(3),rowSds(fig8_sim_stacker[[tag]])/sqrt(3),
                      rowSds(fig8_sim_stu[[tag]])/sqrt(3),
                      rowSds(fig8_sim_stalign[[tag]])/sqrt(3),
                      rowSds(fig8_sim_unaligned_sub0.1[[tag]])/sqrt(3),rowSds(fig8_sim_stacker_sub0.1[[tag]])/sqrt(3),
                      rowSds(fig8_sim_stu_sub0.1[[tag]])/sqrt(3),
                      rowSds(fig8_sim_paste_sub0.1[[tag]])/sqrt(3),
                      rowSds(fig8_sim_stalign_sub0.1[[tag]])/sqrt(3)),
               'Method'=c(rep('Unaligned',nrow(fig8_sim_unaligned[[tag]])),rep('STaCker', nrow(fig8_sim_stacker[[tag]])),
                          rep('STUtiltiy',nrow(fig8_sim_stu[[tag]])),rep('STAlign', nrow(fig8_sim_stalign[[tag]])),
                          rep('Unaligned.sub',nrow(fig8_sim_unaligned_sub0.1[[tag]])),rep('STaCker.sub', nrow(fig8_sim_stacker_sub0.1[[tag]])),
                          rep('STUtiltiy.sub',nrow(fig8_sim_stu_sub0.1[[tag]])),rep('PASTE.sub', nrow(fig8_sim_paste_sub0.1[[tag]])),
                          rep('STAlign.sub',nrow(fig8_sim_stalign_sub0.1[[tag]])))
)
df3$Method=factor(df3$Method, level=c(unique(df3$Method)),order=T)
df3$Gene=factor(df3$Gene, level=c(selg,setdiff(df3$Gene, selg)),order=T)
df3$resolution='200'

tag='pseudo500'
df4=data.frame('Similarity'=c(rowMeans(fig8_sim_unaligned[[tag]],na.rm=T),
                              rowMeans(fig8_sim_stacker[[tag]],na.rm=T),
                              rowMeans(fig8_sim_stu[[tag]],na.rm=T),
                              rowMeans(fig8_sim_stalign[[tag]],na.rm=T),
                              rowMeans(fig8_sim_unaligned_sub0.1[[tag]],na.rm=T),
                              rowMeans(fig8_sim_stacker_sub0.1[[tag]],na.rm=T),
                              rowMeans(fig8_sim_stu_sub0.1[[tag]],na.rm=T),
                              rowMeans(fig8_sim_paste_sub0.1[[tag]],na.rm=T),
                              rowMeans(fig8_sim_stalign_sub0.1[[tag]],na.rm=T)),
               'Gene'=c(rownames(fig8_sim_unaligned[[tag]]), rownames(fig8_sim_stacker[[tag]]),
                        rownames(fig8_sim_stu[[tag]]), rownames(fig8_sim_stalign[[tag]]),
                        rownames(fig8_sim_unaligned_sub0.1[[tag]]), rownames(fig8_sim_stacker_sub0.1[[tag]]),
                        rownames(fig8_sim_stu_sub0.1[[tag]]), rownames(fig8_sim_paste_sub0.1[[tag]]),
                        rownames(fig8_sim_stalign_sub0.1[[tag]])),
               'SE'=c(rowSds(fig8_sim_unaligned[[tag]])/sqrt(3),rowSds(fig8_sim_stacker[[tag]])/sqrt(3),
                      rowSds(fig8_sim_stu[[tag]])/sqrt(3),
                      rowSds(fig8_sim_stalign[[tag]])/sqrt(3),
                      rowSds(fig8_sim_unaligned_sub0.1[[tag]])/sqrt(3),rowSds(fig8_sim_stacker_sub0.1[[tag]])/sqrt(3),
                      rowSds(fig8_sim_stu_sub0.1[[tag]])/sqrt(3),
                      rowSds(fig8_sim_paste_sub0.1[[tag]])/sqrt(3),
                      rowSds(fig8_sim_stalign_sub0.1[[tag]])/sqrt(3)),
               'Method'=c(rep('Unaligned',nrow(fig8_sim_unaligned[[tag]])),rep('STaCker', nrow(fig8_sim_stacker[[tag]])),
                          rep('STUtiltiy',nrow(fig8_sim_stu[[tag]])),rep('STAlign', nrow(fig8_sim_stalign[[tag]])),
                          rep('Unaligned.sub',nrow(fig8_sim_unaligned_sub0.1[[tag]])),rep('STaCker.sub', nrow(fig8_sim_stacker_sub0.1[[tag]])),
                          rep('STUtiltiy.sub',nrow(fig8_sim_stu_sub0.1[[tag]])),rep('PASTE.sub', nrow(fig8_sim_paste_sub0.1[[tag]])),
                          rep('STAlign.sub',nrow(fig8_sim_stalign_sub0.1[[tag]])))
)
df4$Method=factor(df4$Method, level=c(unique(df4$Method)),order=T)
df4$Gene=factor(df4$Gene, level=c(selg,setdiff(df4$Gene, selg)),order=T)
df4$resolution='500'

df=rbind(df1,df2,df3,df4)
df$resolution=factor(df$resolution,level=c(50,100,200,500),order=T)


#
tag='pseudo50'
pdf(paste0(rootdir,'figures/Fig8_selGene_Similarity_',tag,'.pdf'),width=2.5,height=1.8,pointsize = 10)
ggplot(df[which(df$Method %in% c("Unaligned",'STaCker','STUtiltiy','STAlign') &df$resolution==gsub('pseudo','',tag) & df$Gene %in% selg ),],aes(x=Gene, y=Similarity, group=Method, fill=Method))+geom_bar(stat="identity",position='dodge',show.legend = F)+
  geom_errorbar(aes(ymin=Similarity-SE, ymax=Similarity+SE), position =  position_dodge(width = 0.9), width=0.1, colour="black", alpha=1, size=0.2)+
  scale_colour_manual(values=c('grey','red','green2','blue','black','orange','aquamarine2','violet','cyan','gold'))+
  scale_fill_manual(values=c('grey','red','green2','blue','black','orange','aquamarine2','violet','cyan','gold'))+
  ylim(0,0.6)+
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black"),
        axis.title.x=element_blank(),#axis.text.y=element_blank(), 
        axis.title.y=element_blank(),axis.text.x = element_text(angle = 0, vjust = 1, hjust=0.5))
dev.off()

pdf(paste0(rootdir,'figures/suppFig10_selGene_Similarity_subsampled_',tag,'.pdf'),width=2.5,height=1.8,pointsize = 10)
ggplot(df[which(df$Method %in% c("Unaligned.sub",'STaCker.sub','STUtiltiy.sub','PASTE.sub','STAlign.sub') &df$resolution==gsub('pseudo','',tag) & df$Gene %in% selg),],aes(x=Gene, y=Similarity, group=Method, fill=Method))+geom_bar(stat="identity",position='dodge',show.legend = F)+
  geom_errorbar(aes(ymin=Similarity-SE, ymax=Similarity+SE), position =  position_dodge(width = 0.9), width=0.1, colour="black", alpha=1, size=0.2)+
  scale_colour_manual(values=c('grey','red','green2','blue','black','orange','aquamarine2','violet','cyan','gold'))+
  scale_fill_manual(values=c('grey','red','green2','blue','black','orange','aquamarine2','violet','cyan','gold'))+
  ylim(0,0.35)+
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black"),
        axis.title.x=element_blank(),#axis.text.y=element_blank(), 
        axis.title.y=element_blank(),axis.text.x = element_text(angle = 0, vjust = 1, hjust=0.5))
dev.off()

pdf(paste0(rootdir,'figures/Fig8_nonTrivialGenes_similarity_',tag,'.pdf'),width=2.,height=1.8,pointsize = 20)
ggplot(df[which(df$Method %in% c("Unaligned",'STaCker','STUtiltiy','STAlign')&df$resolution==gsub('pseudo','',tag) & df$Gene %in% nontrivial[[tag]]),],aes(x=Method, y=Similarity, group=Method, color=Method))+
  geom_boxplot(show.legend = FALSE,width = 0.5)+
  scale_colour_manual(values=c('grey','red','green2','blue','black','orange','aquamarine2','violet','cyan','gold'))+
  scale_fill_manual(values=c('grey','red','green2','blue','black','orange','aquamarine2','violet','cyan','gold'))+
  ylim(0.0,0.8)+coord_cartesian(ylim=c(0.0,0.7))+
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black"),
        axis.title.x=element_blank(),#axis.text.y=element_blank(), 
        axis.title.y=element_blank(),axis.text.x=element_blank())#axis.text.x = element_text(angle = 45, vjust = 1, hjust=1))
dev.off()

pdf(paste0(rootdir,'figures/suppFig10_nonTrivialGenes_similarity_subsampled_',tag,'.pdf'),width=2.,height=1.8,pointsize = 20)
ggplot(df[which(df$Method %in% c("Unaligned.sub",'STaCker.sub','STUtiltiy.sub','PASTE.sub','STAlign.sub')&df$resolution==gsub('pseudo','',tag) & df$Gene %in% nontrivial_sub0.1[[tag]]),],aes(x=Method, y=Similarity, group=Method, color=Method))+geom_boxplot(show.legend = FALSE)+
  scale_colour_manual(values=c('grey','red','green2','blue','black','orange','aquamarine2','violet','cyan','gold'))+
  scale_fill_manual(values=c('grey','red','green2','blue','black','orange','aquamarine2','violet','cyan','gold'))+
  ylim(0.0,0.75)+coord_cartesian(ylim=c(0.0,0.6))+
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black"),
        axis.title.x=element_blank(),#axis.text.y=element_blank(), 
        axis.title.y=element_blank(),axis.text.x=element_blank())#axis.text.x = element_text(angle = 45, vjust = 1, hjust=1))
dev.off()

tag='pseudo100'
#tag='pseudo200'
#tag='pseudo500'
pdf(paste0(rootdir,'figures/suppFig9_selGene_Similarity_',tag,'.pdf'),width=2.5,height=1.8,pointsize = 10)
ggplot(df[which(df$Method %in% c("Unaligned",'STaCker','STUtiltiy','STAlign') &df$resolution==gsub('pseudo','',tag) & df$Gene %in% selg ),],aes(x=Gene, y=Similarity, group=Method, fill=Method))+geom_bar(stat="identity",position='dodge',show.legend = F)+
  geom_errorbar(aes(ymin=Similarity-SE, ymax=Similarity+SE), position =  position_dodge(width = 0.9), width=0.1, colour="black", alpha=1, size=0.2)+
  scale_colour_manual(values=c('grey','red','green2','blue','black','orange','aquamarine2','violet','cyan','gold'))+
  scale_fill_manual(values=c('grey','red','green2','blue','black','orange','aquamarine2','violet','cyan','gold'))+
  ylim(0,1)+
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black"),
        axis.title.x=element_blank(),#axis.text.y=element_blank(), 
        axis.title.y=element_blank(),axis.text.x = element_text(angle = 0, vjust = 1, hjust=0.5))
dev.off()

pdf(paste0(rootdir,'figures/suppFig11_selGene_Similarity_subsampled_',tag,'.pdf'),width=2.5,height=1.8,pointsize = 10)
ggplot(df[which(df$Method %in% c("Unaligned.sub",'STaCker.sub','STUtiltiy.sub','PASTE.sub','STAlign.sub') &df$resolution==gsub('pseudo','',tag) & df$Gene %in% selg),],aes(x=Gene, y=Similarity, group=Method, fill=Method))+geom_bar(stat="identity",position='dodge',show.legend = F)+
  geom_errorbar(aes(ymin=Similarity-SE, ymax=Similarity+SE), position =  position_dodge(width = 0.9), width=0.1, colour="black", alpha=1, size=0.2)+
  scale_colour_manual(values=c('grey','red','green2','blue','black','orange','aquamarine2','violet','cyan','gold'))+
  scale_fill_manual(values=c('grey','red','green2','blue','black','orange','aquamarine2','violet','cyan','gold'))+
  ylim(0,0.9)+
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black"),
        axis.title.x=element_blank(),#axis.text.y=element_blank(), 
        axis.title.y=element_blank(),axis.text.x = element_text(angle = 0, vjust = 1, hjust=0.5))
dev.off()

pdf(paste0(rootdir,'figures/suppFig9_nonTrivialGenes_similarity_',tag,'.pdf'),width=2.,height=1.8,pointsize = 20)
ggplot(df[which(df$Method %in% c("Unaligned",'STaCker','STUtiltiy','STAlign')&df$resolution==gsub('pseudo','',tag) & df$Gene %in% nontrivial[[tag]]),],aes(x=Method, y=Similarity, group=Method, color=Method))+geom_boxplot(show.legend = FALSE)+
  scale_colour_manual(values=c('grey','red','green2','blue','black','orange','aquamarine2','violet','cyan','gold'))+
  scale_fill_manual(values=c('grey','red','green2','blue','black','orange','aquamarine2','violet','cyan','gold'))+
  ylim(0.0,1)+
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black"),
        axis.title.x=element_blank(),#axis.text.y=element_blank(), 
        axis.title.y=element_blank(),axis.text.x=element_blank())#axis.text.x = element_text(angle = 45, vjust = 1, hjust=1))
dev.off()

pdf(paste0(rootdir,'figures/suppFig11_nonTrivialGenes_similarity_subsampled_',tag,'.pdf'),width=2.,height=1.8,pointsize = 20)
ggplot(df[which(df$Method %in% c("Unaligned.sub",'STaCker.sub','STUtiltiy.sub','PASTE.sub','STAlign.sub')&df$resolution==gsub('pseudo','',tag) & df$Gene %in% nontrivial_sub0.1[[tag]]),],aes(x=Method, y=Similarity, group=Method, color=Method))+geom_boxplot(show.legend = FALSE)+
  scale_colour_manual(values=c('grey','red','green2','blue','black','orange','aquamarine2','violet','cyan','gold'))+
  scale_fill_manual(values=c('grey','red','green2','blue','black','orange','aquamarine2','violet','cyan','gold'))+
  ylim(0.0,1)+
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black"),
        axis.title.x=element_blank(),#axis.text.y=element_blank(), 
        axis.title.y=element_blank(),axis.text.x=element_blank())#axis.text.x = element_text(angle = 45, vjust = 1, hjust=1))
dev.off()


## PseudoSpot views
mySpatialDimPlot2<-function(coords,exps,g,magnify=1000,qfactor=1){
  colnames(coords)=c('x','y')
  df=as.data.frame(coords)
  df$exp=exps[g,rownames(df)]
  df$x=df$x*magnify
  df$y=df$y*magnify
  df$slice=as.numeric(as.factor(gsub(".*:",'',rownames(df))))
  df$exp1=df$exp
  df$exp1[df$slice!=1]=0
  df$exp2=df$exp
  df$exp2[df$slice!=2]=0
  df$rgb=(rgb(df$exp1/max(df$exp1), 0, df$exp2/max(df$exp2)))
  p1=ggplot(df,aes(x,-y))+
    geom_tile(alpha=df$exp1/quantile(df$exp1,qfactor), fill="red")+
    geom_tile(alpha=df$exp2/quantile(df$exp2,qfactor), fill="blue")+
    theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
          panel.background = element_blank(),
          axis.title.x=element_blank(),axis.text.x=element_blank(),axis.ticks.x=element_blank(),
          axis.title.y=element_blank(),axis.text.y=element_blank(),axis.ticks.y=element_blank()
    )
  p1
}

tag='pseudo50'
g='Htr1a'
p4=mySpatialDimPlot2(fig8_ps_coords_stalign[['ref0']][[paste0(tag,'_combopos')]][grep(':(1|2)',rownames(fig8_ps_coords_stalign[['ref0']][[paste0(tag,'_combopos')]])),],
                     fig8_ps_coords_stalign[['ref0']][[paste0(tag,'_comboexp')]][,grep(':(1|2)',colnames(fig8_ps_coords_stalign[['ref0']][[paste0(tag,'_comboexp')]]))],g=g)
p3=mySpatialDimPlot2(fig8_ps_coords_stacker[[paste0(tag,'_combopos')]][grep(':(1|2)',rownames(fig8_ps_coords_stacker[[paste0(tag,'_combopos')]])),],
                     fig8_ps_coords_stacker[[paste0(tag,'_comboexp')]][,grep(':(1|2)',colnames(fig8_ps_coords_stacker[[paste0(tag,'_comboexp')]]))],g=g)
p2=mySpatialDimPlot2(fig8_ps_coords_stu[['ref0']][[paste0(tag,'_combopos')]][grep(':(1|2)',rownames(fig8_ps_coords_stu[['ref0']][[paste0(tag,'_combopos')]])),],
                     fig8_ps_coords_stu[['ref0']][[paste0(tag,'_comboexp')]][,grep(':(1|2)',colnames(fig8_ps_coords_stu[['ref0']][[paste0(tag,'_comboexp')]]))],g=g)
pdf(paste0(rootdir,'figures/suppFig8_Pseudospot_',gsub("pseudo",'',tag),'_',g,'.pdf'),width=3.7,height=10,pointsize = 10)
ggarrange(p2,p3,p4,ncol=1)
dev.off()

g='Efemp1';
#g='Drd1'
#g='Gpr101'
p4=mySpatialDimPlot2(fig8_ps_coords_stalign[['ref0']][[paste0(tag,'_combopos')]][grep(':(1|2)',rownames(fig8_ps_coords_stalign[['ref0']][[paste0(tag,'_combopos')]])),],
                  fig8_ps_coords_stalign[['ref0']][[paste0(tag,'_comboexp')]][,grep(':(1|2)',colnames(fig8_ps_coords_stalign[['ref0']][[paste0(tag,'_comboexp')]]))],g=g,qfactor=0.999)
p3=mySpatialDimPlot2(fig8_ps_coords_stacker[[paste0(tag,'_combopos')]][grep(':(1|2)',rownames(fig8_ps_coords_stacker[[paste0(tag,'_combopos')]])),],
                  fig8_ps_coords_stacker[[paste0(tag,'_comboexp')]][,grep(':(1|2)',colnames(fig8_ps_coords_stacker[[paste0(tag,'_comboexp')]]))],g=g,qfactor=0.999)
p2=mySpatialDimPlot2(fig8_ps_coords_stu[['ref0']][[paste0(tag,'_combopos')]][grep(':(1|2)',rownames(fig8_ps_coords_stu[['ref0']][[paste0(tag,'_combopos')]])),],
                  fig8_ps_coords_stu[['ref0']][[paste0(tag,'_comboexp')]][,grep(':(1|2)',colnames(fig8_ps_coords_stu[['ref0']][[paste0(tag,'_comboexp')]]))],g=g,qfactor=0.999)
pdf(paste0(rootdir,'figures/suppFig8_Pseudospot_',gsub("pseudo",'',tag),'_',g,'.pdf'),width=3.7,height=10,pointsize = 10)
ggarrange(p2,p3,p4,ncol=1)
dev.off()

#
tag='pseudo100'
g='Htr1a'
p4=mySpatialDimPlot2(fig8_ps_coords_stalign[['ref0']][[paste0(tag,'_combopos')]][grep(':(1|2)',rownames(fig8_ps_coords_stalign[['ref0']][[paste0(tag,'_combopos')]])),],
                     fig8_ps_coords_stalign[['ref0']][[paste0(tag,'_comboexp')]][,grep(':(1|2)',colnames(fig8_ps_coords_stalign[['ref0']][[paste0(tag,'_comboexp')]]))],g=g)
p3=mySpatialDimPlot2(fig8_ps_coords_stacker[[paste0(tag,'_combopos')]][grep(':(1|2)',rownames(fig8_ps_coords_stacker[[paste0(tag,'_combopos')]])),],
                     fig8_ps_coords_stacker[[paste0(tag,'_comboexp')]][,grep(':(1|2)',colnames(fig8_ps_coords_stacker[[paste0(tag,'_comboexp')]]))],g=g)
p2=mySpatialDimPlot2(fig8_ps_coords_stu[['ref0']][[paste0(tag,'_combopos')]][grep(':(1|2)',rownames(fig8_ps_coords_stu[['ref0']][[paste0(tag,'_combopos')]])),],
                     fig8_ps_coords_stu[['ref0']][[paste0(tag,'_comboexp')]][,grep(':(1|2)',colnames(fig8_ps_coords_stu[['ref0']][[paste0(tag,'_comboexp')]]))],g=g)
pdf(paste0(rootdir,'figures/suppFig8_Pseudospot_',gsub("pseudo",'',tag),'_',g,'.pdf'),width=3.7,height=10,pointsize = 10)
ggarrange(p2,p3,p4,ncol=1)
dev.off()
g='Efemp1';
#g='Drd1'
#g='Gpr101'
p4=mySpatialDimPlot2(fig8_ps_coords_stalign[['ref0']][[paste0(tag,'_combopos')]][grep(':(1|2)',rownames(fig8_ps_coords_stalign[['ref0']][[paste0(tag,'_combopos')]])),],
                     fig8_ps_coords_stalign[['ref0']][[paste0(tag,'_comboexp')]][,grep(':(1|2)',colnames(fig8_ps_coords_stalign[['ref0']][[paste0(tag,'_comboexp')]]))],g=g,qfactor=0.9995)
p3=mySpatialDimPlot2(fig8_ps_coords_stacker[[paste0(tag,'_combopos')]][grep(':(1|2)',rownames(fig8_ps_coords_stacker[[paste0(tag,'_combopos')]])),],
                     fig8_ps_coords_stacker[[paste0(tag,'_comboexp')]][,grep(':(1|2)',colnames(fig8_ps_coords_stacker[[paste0(tag,'_comboexp')]]))],g=g,qfactor=0.9995)
p2=mySpatialDimPlot2(fig8_ps_coords_stu[['ref0']][[paste0(tag,'_combopos')]][grep(':(1|2)',rownames(fig8_ps_coords_stu[['ref0']][[paste0(tag,'_combopos')]])),],
                     fig8_ps_coords_stu[['ref0']][[paste0(tag,'_comboexp')]][,grep(':(1|2)',colnames(fig8_ps_coords_stu[['ref0']][[paste0(tag,'_comboexp')]]))],g=g,qfactor=0.9995)
pdf(paste0(rootdir,'figures/suppFig8_Pseudospot_',gsub("pseudo",'',tag),'_',g,'.pdf'),width=3.7,height=10,pointsize = 10)
ggarrange(p2,p3,p4,ncol=1)
dev.off()

#
tag='pseudo200'
g='Htr1a'
p4=mySpatialDimPlot2(fig8_ps_coords_stalign[['ref0']][[paste0(tag,'_combopos')]][grep(':(1|2)',rownames(fig8_ps_coords_stalign[['ref0']][[paste0(tag,'_combopos')]])),],
                     fig8_ps_coords_stalign[['ref0']][[paste0(tag,'_comboexp')]][,grep(':(1|2)',colnames(fig8_ps_coords_stalign[['ref0']][[paste0(tag,'_comboexp')]]))],g=g)
p3=mySpatialDimPlot2(fig8_ps_coords_stacker[[paste0(tag,'_combopos')]][grep(':(1|2)',rownames(fig8_ps_coords_stacker[[paste0(tag,'_combopos')]])),],
                     fig8_ps_coords_stacker[[paste0(tag,'_comboexp')]][,grep(':(1|2)',colnames(fig8_ps_coords_stacker[[paste0(tag,'_comboexp')]]))],g=g)
p2=mySpatialDimPlot2(fig8_ps_coords_stu[['ref0']][[paste0(tag,'_combopos')]][grep(':(1|2)',rownames(fig8_ps_coords_stu[['ref0']][[paste0(tag,'_combopos')]])),],
                     fig8_ps_coords_stu[['ref0']][[paste0(tag,'_comboexp')]][,grep(':(1|2)',colnames(fig8_ps_coords_stu[['ref0']][[paste0(tag,'_comboexp')]]))],g=g)
pdf(paste0(rootdir,'figures/suppFig8_Pseudospot_',gsub("pseudo",'',tag),'_',g,'.pdf'),width=3.7,height=10,pointsize = 10)
ggarrange(p2,p3,p4,ncol=1)
dev.off()
g='Efemp1';
#g='Drd1'
#g='Gpr101'
p4=mySpatialDimPlot2(fig8_ps_coords_stalign[['ref0']][[paste0(tag,'_combopos')]][grep(':(1|2)',rownames(fig8_ps_coords_stalign[['ref0']][[paste0(tag,'_combopos')]])),],
                     fig8_ps_coords_stalign[['ref0']][[paste0(tag,'_comboexp')]][,grep(':(1|2)',colnames(fig8_ps_coords_stalign[['ref0']][[paste0(tag,'_comboexp')]]))],g=g)
p3=mySpatialDimPlot2(fig8_ps_coords_stacker[[paste0(tag,'_combopos')]][grep(':(1|2)',rownames(fig8_ps_coords_stacker[[paste0(tag,'_combopos')]])),],
                     fig8_ps_coords_stacker[[paste0(tag,'_comboexp')]][,grep(':(1|2)',colnames(fig8_ps_coords_stacker[[paste0(tag,'_comboexp')]]))],g=g)
p2=mySpatialDimPlot2(fig8_ps_coords_stu[['ref0']][[paste0(tag,'_combopos')]][grep(':(1|2)',rownames(fig8_ps_coords_stu[['ref0']][[paste0(tag,'_combopos')]])),],
                     fig8_ps_coords_stu[['ref0']][[paste0(tag,'_comboexp')]][,grep(':(1|2)',colnames(fig8_ps_coords_stu[['ref0']][[paste0(tag,'_comboexp')]]))],g=g)
pdf(paste0(rootdir,'figures/suppFig8_Pseudospot_',gsub("pseudo",'',tag),'_',g,'.pdf'),width=3.7,height=10,pointsize = 10)
ggarrange(p2,p3,p4,ncol=1)
dev.off()

tag='pseudo500'
g='Htr1a'
p4=mySpatialDimPlot2(fig8_ps_coords_stalign[['ref0']][[paste0(tag,'_combopos')]][grep(':(1|2)',rownames(fig8_ps_coords_stalign[['ref0']][[paste0(tag,'_combopos')]])),],
                     fig8_ps_coords_stalign[['ref0']][[paste0(tag,'_comboexp')]][,grep(':(1|2)',colnames(fig8_ps_coords_stalign[['ref0']][[paste0(tag,'_comboexp')]]))],g=g)
p3=mySpatialDimPlot2(fig8_ps_coords_stacker[[paste0(tag,'_combopos')]][grep(':(1|2)',rownames(fig8_ps_coords_stacker[[paste0(tag,'_combopos')]])),],
                     fig8_ps_coords_stacker[[paste0(tag,'_comboexp')]][,grep(':(1|2)',colnames(fig8_ps_coords_stacker[[paste0(tag,'_comboexp')]]))],g=g)
p2=mySpatialDimPlot2(fig8_ps_coords_stu[['ref0']][[paste0(tag,'_combopos')]][grep(':(1|2)',rownames(fig8_ps_coords_stu[['ref0']][[paste0(tag,'_combopos')]])),],
                     fig8_ps_coords_stu[['ref0']][[paste0(tag,'_comboexp')]][,grep(':(1|2)',colnames(fig8_ps_coords_stu[['ref0']][[paste0(tag,'_comboexp')]]))],g=g)
pdf(paste0(rootdir,'figures/suppFig8_Pseudospot_',gsub("pseudo",'',tag),'_',g,'.pdf'),width=3.7,height=10,pointsize = 10)
ggarrange(p2,p3,p4,ncol=1)
dev.off()
g='Efemp1';
#g='Drd1'
#g='Gpr101'
p4=mySpatialDimPlot2(fig8_ps_coords_stalign[['ref0']][[paste0(tag,'_combopos')]][grep(':(1|2)',rownames(fig8_ps_coords_stalign[['ref0']][[paste0(tag,'_combopos')]])),],
                     fig8_ps_coords_stalign[['ref0']][[paste0(tag,'_comboexp')]][,grep(':(1|2)',colnames(fig8_ps_coords_stalign[['ref0']][[paste0(tag,'_comboexp')]]))],g=g)
p3=mySpatialDimPlot2(fig8_ps_coords_stacker[[paste0(tag,'_combopos')]][grep(':(1|2)',rownames(fig8_ps_coords_stacker[[paste0(tag,'_combopos')]])),],
                     fig8_ps_coords_stacker[[paste0(tag,'_comboexp')]][,grep(':(1|2)',colnames(fig8_ps_coords_stacker[[paste0(tag,'_comboexp')]]))],g=g)
p2=mySpatialDimPlot2(fig8_ps_coords_stu[['ref0']][[paste0(tag,'_combopos')]][grep(':(1|2)',rownames(fig8_ps_coords_stu[['ref0']][[paste0(tag,'_combopos')]])),],
                     fig8_ps_coords_stu[['ref0']][[paste0(tag,'_comboexp')]][,grep(':(1|2)',colnames(fig8_ps_coords_stu[['ref0']][[paste0(tag,'_comboexp')]]))],g=g)
pdf(paste0(rootdir,'figures/suppFig8_Pseudospot_',gsub("pseudo",'',tag),'_',g,'.pdf'),width=3.7,height=10,pointsize = 10)
ggarrange(p2,p3,p4,ncol=1)
dev.off()

## volcano plot
tag='pseudo50'
df=data.frame('fc'=fig8_moran_stacker[[tag]][nontrivial[[tag]],1]/rowMeans(fig8_moran_stalign[[tag]][nontrivial[[tag]],]),
  'pval'=sapply(nontrivial[[tag]], function(x){t.test((fig8_moran_stalign[[tag]][x,]),mu=fig8_moran_stacker[[tag]][x,1])$p.value})
    )
diffexpress=rep('up',length(nontrivial[[tag]]))
diffexpress[which(df$fc<1)]='down'
diffexpress[which(df$pval>0.05)]='ns'
df$diffexpress=factor(diffexpress, level=c('down','ns','up'),order=T)
pdf(paste0(rootdir,'figures/Fig8_volcano_moran.pdf'), width=2., height=2,pointsize = 10)
ggplot(data = df, aes(x = log2(fc), y = -log10(pval), col = diffexpress)) +
  geom_hline(yintercept = -log10(0.05), col = "gray", linetype = 'dashed') +
  geom_point(size = 1) +
  scale_color_manual(values = c("blue", "grey", "orange")) + # to set the labels in case we want to overwrite the categories from the dataframe (UP, DOWN, NO)<br />
  coord_cartesian(ylim = c(0, 4), xlim = c(-0.2, 0.2)) +
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black"),
        legend.position="none")
#theme_bw()
dev.off()

tag='pseudo50'
df=data.frame('fc'=rowMeans(fig8_sim_stacker[[tag]][nontrivial[[tag]],])/rowMeans(fig8_sim_stalign[[tag]][nontrivial[[tag]],]),
              'pval'=sapply(nontrivial[[tag]], function(x){t.test((fig8_sim_stacker[[tag]][x,]),fig8_sim_stalign[[tag]][x,])$p.value})
)
diffexpress=rep('up',length(nontrivial[[tag]]))
diffexpress[which(df$fc<1)]='down'
diffexpress[which(df$pval>0.05)]='ns'
df$diffexpress=factor(diffexpress, level=c('down','ns','up'),order=T)
pdf(paste0(rootdir,'figures/Fig8_volcano_Sim.pdf'), width=2., height=2,pointsize = 10)
ggplot(data = df, aes(x = log2(fc), y = -log10(pval), col = diffexpress)) +
  geom_hline(yintercept = -log10(0.05), col = "gray", linetype = 'dashed') +
  geom_point(size = 1) +
  scale_color_manual(values = c("grey", "orange")) + # to set the labels in case we want to overwrite the categories from the dataframe (UP, DOWN, NO)<br />
  coord_cartesian(ylim = c(0, 4), xlim = c(-0.48, 0.48)) +
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black"),
        legend.position="none")
dev.off()


tag='pseudo50'
df=data.frame('fc'=fig8_moran_stacker_sub0.1[[tag]][nontrivial_sub0.1[[tag]],1]/rowMeans(fig8_moran_stalign_sub0.1[[tag]][nontrivial_sub0.1[[tag]],]),
              'pval'=sapply(nontrivial_sub0.1[[tag]], function(x){t.test((fig8_moran_stalign_sub0.1[[tag]][x,]),mu=fig8_moran_stacker_sub0.1[[tag]][x,1])$p.value})
)
diffexpress=rep('up',length(nontrivial_sub0.1[[tag]]))
diffexpress[which(df$fc<1)]='down'
diffexpress[which(df$pval>0.05)]='ns'
df$diffexpress=factor(diffexpress, level=c('down','ns','up'),order=T)
pdf(paste0(rootdir,'figures/suppFig10_volcano_moran.pdf'), width=2, height=2.,pointsize = 8)
ggplot(data = df, aes(x = log2(fc), y = -log10(pval), col = diffexpress)) +
  geom_hline(yintercept = -log10(0.05), col = "gray", linetype = 'dashed') +
  geom_point(size = 0.8) +
  scale_color_manual(values = c("blue", "grey", "orange")) + # to set the labels in case we want to overwrite the categories from the dataframe (UP, DOWN, NO)<br />
  coord_cartesian(ylim = c(0, 5.5), xlim = c(-0.5, 0.5)) +
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black"),
        legend.position="none")
dev.off()

tag='pseudo50'
df=data.frame('fc'=rowMeans(fig8_sim_stacker_sub0.1[[tag]][nontrivial_sub0.1[[tag]],])/rowMeans(fig8_sim_stalign_sub0.1[[tag]][nontrivial_sub0.1[[tag]],]),
              'pval'=sapply(nontrivial_sub0.1[[tag]], function(x){
                if(length(setdiff(fig8_sim_stalign_sub0.1[[tag]][x,],NaN))>1 & 
                   length(setdiff(fig8_sim_stacker_sub0.1[[tag]][x,],NaN))>1){
                  t.test((fig8_sim_stacker_sub0.1[[tag]][x,]),fig8_sim_stalign_sub0.1[[tag]][x,])$p.value}else{1}})
)
df$fc[which(!is.finite(df$fc))]=1
diffexpress=rep('up',length(nontrivial_sub0.1[[tag]]))
diffexpress[which(df$fc<1)]='down'
diffexpress[which(df$pval>0.05)]='ns'
df$diffexpress=factor(diffexpress, level=c('down','ns','up'),order=T)
pdf(paste0(rootdir,'figures/suppFig10_volcano_Sim.pdf'), width=2., height=2.,pointsize = 10)
ggplot(data = df, aes(x = log2(fc), y = -log10(pval), col = diffexpress)) +
  geom_hline(yintercept = -log10(0.05), col = "gray", linetype = 'dashed') +
  geom_point(size = 0.8) +
  scale_color_manual(values = c('blue',"grey", "orange")) + # to set the labels in case we want to overwrite the categories from the dataframe (UP, DOWN, NO)<br />
  coord_cartesian(ylim = c(0, 4), xlim = c(-1.6, 1.6)) +
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black"),
        legend.position="none")
dev.off()


####################### Fig9
load(paste0(rootdir,'data/Fig9_metrics.rda'))

# unaligned
df=data.frame('x'=c(fig9_coords_unaligned[['v']][,1],fig9_coords_unaligned[['x']][,1]),
              'y'=c(fig9_coords_unaligned[['v']][,2],fig9_coords_unaligned[['x']][,2]),
              'nCounts_v'=c(rep(1,nrow(fig9_coords_unaligned[['v']])),
                            rep(0,nrow(fig9_coords_unaligned[['x']]))),
              'nCounts_x'=c(rep(0,nrow(fig9_coords_unaligned[['v']])),
                            rep(1,nrow(fig9_coords_unaligned[['x']]))),
              'group'=c(rep('v',times=nrow(fig9_coords_unaligned[['v']])),rep('x',times=nrow(fig9_coords_unaligned[['x']])))
)
pdf(paste0(rootdir,'/figures/Fig9_spatial_unaligned.pdf'),width=2.5,height=3.1,pointsize = 1)
ggplot(df,aes(x,-y))+
  geom_point(alpha=df$nCounts_x/max(df$nCounts_x),col='blue',fill="blue",size=0.1,shape=4,stroke=0.1)+
  geom_point(alpha=(df$nCounts_v),col='red',fill='red', size=1,stroke=0.001)+#shape=1,
  theme_void()
dev.off()

#stacker
df=data.frame('x'=c(fig9_coords_stacker[['v']][,1],fig9_coords_stacker[['x']][,1]),
              'y'=c(fig9_coords_stacker[['v']][,2],fig9_coords_stacker[['x']][,2]),
              'nCounts_v'=c(rep(0.5,nrow(fig9_coords_stacker[['v']])),
                            rep(0,nrow(fig9_coords_stacker[['x']]))),
              'nCounts_x'=c(rep(0,nrow(fig9_coords_stacker[['v']])),
                            rep(1,nrow(fig9_coords_stacker[['x']]))),
              'group'=c(rep('v',times=nrow(fig9_coords_stacker[['v']])),rep('x',times=nrow(fig9_coords_stacker[['x']])))
)
pdf(paste0(rootdir,'/figures/Fig9_spatial_STACKER.pdf'),width=2.5,height=3.1,pointsize = 1)
ggplot(df,aes(x,-y))+
  geom_point(alpha=df$nCounts_x/max(df$nCounts_x),col='blue',fill="blue",size=0.1,shape=4,stroke=0.1)+
  geom_point(alpha=(df$nCounts_v),col='red',fill='red', size=1,stroke=0.001)+#shape=1,
  theme_void()
dev.off()

##
df=data.frame('x'=c(fig9_coords_stu[['ref0']][['v']][,1],fig9_coords_stu[['ref0']][['x']][,1]),
              'y'=c(fig9_coords_stu[['ref0']][['v']][,2],fig9_coords_stu[['ref0']][['x']][,2]),
              'nCounts_v'=c(rep(1,nrow(fig9_coords_stu[['ref0']][['v']])),
                            rep(0,nrow(fig9_coords_stu[['ref0']][['x']]))),
              'nCounts_x'=c(rep(0,nrow(fig9_coords_stu[['ref0']][['v']])),
                            rep(1,nrow(fig9_coords_stu[['ref0']][['x']]))),
              'group'=c(rep('v',times=nrow(fig9_coords_stu[['ref0']][['v']])),
                        rep('x',times=nrow(fig9_coords_stu[['ref0']][['x']])))
)
pdf(paste0(rootdir,'/figures/Fig9_spatial_STU.pdf'),width=2.5,height=3.1,pointsize = 1)
ggplot(df,aes(x,-y))+
  geom_point(alpha=df$nCounts_x/max(df$nCounts_x),col='blue',fill="blue",size=0.1,shape=4,stroke=0.1)+
  geom_point(alpha=(df$nCounts_v),col='red',fill='red', size=1.,stroke=0.001)+#stroke=0.001
  theme_void()
dev.off()

#
df=data.frame('x'=c(fig9_coords_stalign[['ref1']][['v']][,1],fig9_coords_stalign[['ref1']][['x']][,1]),
              'y'=c(fig9_coords_stalign[['ref1']][['v']][,2],fig9_coords_stalign[['ref1']][['x']][,2]),
              'nCounts_v'=c(rep(1,nrow(fig9_coords_stalign[['ref1']][['v']])),
                            rep(0,nrow(fig9_coords_stalign[['ref1']][['x']]))),
              'nCounts_x'=c(rep(0,nrow(fig9_coords_stalign[['ref1']][['v']])),
                            rep(1,nrow(fig9_coords_stalign[['ref1']][['x']]))),
              'group'=c(rep('v',times=nrow(fig9_coords_stalign[['ref1']][['v']])),
                        rep('x',times=nrow(fig9_coords_stalign[['ref1']][['x']])))
)
pdf(paste0(rootdir,'/figures/Fig9_spatial_stalign.pdf'),width=2.5,height=3.1,pointsize = 1)
ggplot(df,aes(x,-y))+
  geom_point(alpha=df$nCounts_x/max(df$nCounts_x),col='blue',fill="blue",size=0.1,shape=4,stroke=0.12)+
  geom_point(alpha=(df$nCounts_v),col='red',fill='red', size=1.,stroke=0.001)+#shape=1,
  theme_void()
dev.off()


# gene spatial pattern
mySpatialDimPlot3<-function(coords,exps,g,magnify=1000,midpoint=1,qfactor=1){
  colnames(coords)=c('x','y')
  df=as.data.frame(coords)
  df$exp=exps[g,rownames(df)]
  df$x=round(df$x*magnify,0)
  df$y=round(df$y*magnify,0)
  df$slice=1
  df$slice[grep('ps_',rownames(df))]=2
  df$exp1=df$exp
  df$exp1[df$slice!=1]=0
  df$exp2=df$exp
  df$exp2[df$slice!=2]=0
  p1=ggplot(df,aes(x,-y))+
    geom_tile(alpha=df$exp1/quantile(df$exp1,qfactor), fill="red",width=magnify/5.3,height=magnify/5.3)+
    geom_tile(alpha=df$exp2/quantile(df$exp2,qfactor), fill="blue",width=magnify/5.3,height=magnify/5.3)+
    theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
          panel.background = element_blank(),
          axis.title.x=element_blank(),axis.text.x=element_blank(),axis.ticks.x=element_blank(),
          axis.title.y=element_blank(),axis.text.y=element_blank(),axis.ticks.y=element_blank()
    )
  p1
}

g='Calb2'
#g='Aldh1a2'
#g='Fibcd1'
#g='Prox1'

pdf(paste0(rootdir,'figures/Fig9_',g,'_unaligned.pdf'),width=2,height=2,pointsize = 10)
mySpatialDimPlot3(fig9_ps_coords_unaligned[['combopos']],
                 fig9_ps_coords_unaligned[['comboexp']],g=g)
dev.off()

pdf(paste0(rootdir,'figures/Fig9_',g,'_STaCker.pdf'),width=2,height=2,pointsize = 10)
mySpatialDimPlot3(fig9_ps_coords_stacker[['combopos']],
                    fig9_ps_coords_stacker[['comboexp']],g=g) #midpoint=1,magnify=6.
dev.off()

pdf(paste0(rootdir,'figures/Fig9_',g,'_STUtility.pdf'),width=2,height=2,pointsize = 10)
mySpatialDimPlot3(fig9_ps_coords_stu[['ref0']][['combopos']],
                    fig9_ps_coords_stu[['ref0']][['comboexp']],g=g)
dev.off()

pdf(paste0(rootdir,'figures/Fig9_',g,'_STalign.pdf'),width=2,height=2,pointsize = 10)
mySpatialDimPlot3(fig9_ps_coords_stalign[['ref1']][['combopos']],
                    fig9_ps_coords_stalign[['ref1']][['comboexp']],g=g)#,midpoint=1,magnify=6.
dev.off()



## Moran I
df=data.frame('MoranI'=c((fig9_moran_unaligned[selg,1]),(fig9_moran_stacker[selg]),
                         rowMeans(fig9_moran_stu,na.rm=T)[selg], 
                         rowMeans(fig9_moran_stalign)[selg]
),
'Gene'=rep(selg, times=4),
'Method'=c(rep('Unaligned',length(selg)),rep('STaCker', length(selg)),
           rep('STUtiltiy',length(selg)),rep('STAlign', length(selg))),
'SE'=c(rep(0,length(selg)),rep(0,length(selg)),
       rowSds(fig9_moran_stu[selg,])/sqrt(2),
       rowSds(fig9_moran_stalign[selg,])/(2)
)
)
df$Method=factor(df$Method, level=c(unique(df$Method)),order=T)
df$Gene=factor(df$Gene, level=selg,order=T)
#
pdf(paste0(rootdir,'figures/Fig9_selGene_MoranI.pdf'),width=2.5,height=2,pointsize = 10)
ggplot(df,aes(x=Gene, y=MoranI, group=Method, fill=Method))+geom_bar(stat="identity",position='dodge',show.legend = F)+
  geom_errorbar(aes(ymin=MoranI-SE, ymax=MoranI+SE), position =  position_dodge(width = 0.9), width=0.1, colour="black", alpha=1, size=0.2)+
  scale_colour_manual(values=c('grey','red','green2','blue','black','orange','aquamarine2','violet','cyan','gold'))+
  scale_fill_manual(values=c('grey','red','green2','blue','black','orange','aquamarine2','violet','cyan','gold'))+
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black"),
        axis.title.x=element_blank(),
        axis.title.y=element_blank(),axis.text.x = element_text(angle = 0, vjust = 1, hjust=0.5))
dev.off()


fig9_nontrivial=intersect(rownames(fig9_moran_unaligned_x)[which(fig9_moran_unaligned_x[,'pvalue']<=pcut & fig9_moran_unaligned_x[,'moran']>0)],
                    rownames(fig9_moran_unaligned_v)[which(fig9_moran_unaligned_v[,'pvalue']<=pcut & fig9_moran_unaligned_v[,'moran']>0)])

df=data.frame('MoranI'=c((fig9_moran_unaligned[fig9_nontrivial,1]),(fig9_moran_stacker[fig9_nontrivial]),
                         rowMeans(fig9_moran_stu[fig9_nontrivial,],na.rm=T), 
                         rowMeans(fig9_moran_stalign[fig9_nontrivial,])
),
'Gene'=c(fig9_nontrivial,fig9_nontrivial, 
         fig9_nontrivial, fig9_nontrivial),
'Method'=c(rep('Unaligned',length(fig9_nontrivial)),rep('STaCker', length(fig9_nontrivial)),
           rep('STUtiltiy',length(fig9_nontrivial)),rep('STAlign', length(fig9_nontrivial))),
'SE'=c(rep(0,length(fig9_nontrivial)),rep(0, length(fig9_nontrivial)),
       rowSds(fig9_moran_stu[fig9_nontrivial,])/sqrt(2),
       rowSds(fig9_moran_stalign[fig9_nontrivial,])/sqrt(2)
)
)
df$Method=factor(df$Method, level=c(unique(df$Method)),order=T)
pdf(paste0(rootdir,'figures/Fig9_nonTrivialGenes_MoranI.pdf'),width=1.3,height=2,pointsize = 20)
ggplot(df,aes(x=Method, y=MoranI, group=Method,color=Method))+geom_boxplot(show.legend = FALSE)+
  scale_colour_manual(values=c('grey','red','green2','blue','black','orange','aquamarine2','violet','cyan','gold'))+
  scale_fill_manual(values=c('grey','red','green2','blue','black','orange','aquamarine2','violet','cyan','gold'))+
  coord_cartesian(ylim=c(0,0.65))+
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black"),
        axis.title.x=element_blank(),#axis.text.y=element_blank(), 
        axis.title.y=element_blank(), axis.text.x=element_blank() #axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)
  )
dev.off()

## Similarity
df=data.frame('Similarity'=c(fig9_sim_unaligned[selg,],
                             (fig9_sim_stacker[selg]),
                             rowMeans(fig9_sim_stu[selg,]), 
                             rowMeans(fig9_sim_stalign[selg,])
),
'Gene'=c(selg, selg, selg, selg),
'Method'=c(rep('Unaligned',length(selg)),rep('STaCker', length(selg)),
           rep('STUtiltiy',length(selg)),rep('STAlign', length(selg))),
'SE'=c(rep(0,length(selg)),rep(0,length(selg)),
       rowSds(fig9_sim_stu[selg,])/(2),
       rowSds(fig9_sim_stalign[selg,])/(2)
)
)
df$Method=factor(df$Method, level=c(unique(df$Method)),order=T)
df=df[df$Gene %in% selg,]
df$Gene=factor(df$Gene, level=selg,order=T)
#
pdf(paste0(rootdir,'figures/Fig9_selGene_Similarity.pdf'),width=2.5,height=2,pointsize = 10)
ggplot(df,aes(x=Gene, y=Similarity, group=Method, fill=Method))+geom_bar(stat="identity",position='dodge',show.legend = F)+
  geom_errorbar(aes(ymin=Similarity-SE, ymax=Similarity+SE), position =  position_dodge(width = 0.9), width=0.1, colour="black", alpha=1, size=0.2)+
  scale_colour_manual(values=c('grey','red','green2','blue','black','orange','aquamarine2','violet','cyan','gold'))+
  scale_fill_manual(values=c('grey','red','green2','blue','black','orange','aquamarine2','violet','cyan','gold'))+
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black"),
        axis.title.x=element_blank(),#axis.text.y=element_blank(), 
        axis.title.y=element_blank(),axis.text.x = element_text(angle = 0, vjust = 1, hjust=0.5))
dev.off()


df=data.frame('Similarity'=c(fig9_sim_unaligned[fig9_nontrivial,1],
                             (fig9_sim_stacker[fig9_nontrivial]),
                             rowMeans(fig9_sim_stu[fig9_nontrivial,]), 
                             rowMeans(fig9_sim_stalign[fig9_nontrivial,])
),
'Gene'=c(fig9_nontrivial, fig9_nontrivial, 
         fig9_nontrivial, fig9_nontrivial),
'Method'=c(rep('Unaligned',length(fig9_nontrivial)),rep('STaCker', length(fig9_nontrivial)),
           rep('STUtiltiy',length(fig9_nontrivial)),rep('STAlign', length(fig9_nontrivial)))
)
df$Method=factor(df$Method, level=c(unique(df$Method)),order=T)

pdf(paste0(rootdir,'figures/Fig9_nonTrivialGene_similarity','.pdf'),width=1.3,height=2,pointsize = 20)
ggplot(df,aes(x=Method, y=Similarity, group=Method, color=Method))+geom_boxplot(show.legend = FALSE)+
  scale_colour_manual(values=c('grey','red','green2','blue','black','orange','aquamarine2','violet','cyan','gold'))+
  scale_fill_manual(values=c('grey','red','green2','blue','black','orange','aquamarine2','violet','cyan','gold'))+
  coord_cartesian(ylim=c(0,0.7))+
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black"),
        axis.title.x=element_blank(),#axis.text.y=element_blank(), 
        axis.title.y=element_blank(),axis.text.x=element_blank())#axis.text.x = element_text(angle = 45, vjust = 1, hjust=1))
dev.off()

### volcano plot
df=data.frame('fc'=(fig9_moran_stacker[fig9_nontrivial])/rowMeans(fig9_moran_stalign[fig9_nontrivial,]),
              'pval'=sapply(fig9_nontrivial, function(x){t.test((fig9_moran_stalign[x,]),mu=fig9_moran_stacker[x],var.equal=T)$p.value})
)
diffexpress=rep('up',length(fig9_nontrivial))
diffexpress[which(df$fc<1)]='down'
diffexpress[which(df$pval>0.05)]='ns'
df$diffexpress=factor(diffexpress, level=c('down','ns','up'),order=T)
pdf(paste0('revision/git/figures','/Fig9_volcano_moran.pdf'), width=2, height=2,pointsize = 10)
ggplot(data = df, aes(x = log2(fc), y = -log10(pval), col = diffexpress)) +
  geom_hline(yintercept = -log10(0.05), col = "gray", linetype = 'dashed',size=0.5) +
  geom_point(size = 1) +
  scale_color_manual(values = c("blue", "grey", "orange")) + # to set the labels in case we want to overwrite the categories from the dataframe (UP, DOWN, NO)<br />
  coord_cartesian(ylim = c(0, 3.), xlim = c(-2.3, 3.3)) +
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black"),
        legend.position="none")
#theme_bw()
dev.off()


df=data.frame('fc'=(fig9_sim_stacker[fig9_nontrivial])/rowMeans(fig9_sim_stalign[fig9_nontrivial,]),
              'pval'=sapply(fig9_nontrivial, function(x){if(length(setdiff(fig9_sim_stalign[x,],NaN))<2 | is.nan(fig9_sim_stacker[x])){1}else{t.test((fig9_sim_stalign[x,]),mu=fig9_sim_stacker[x])$p.value}})
)
diffexpress=rep('up',length(fig9_nontrivial))
diffexpress[which(df$fc<1)]='down'
diffexpress[which(df$pval>0.05)]='ns'
df$diffexpress=factor(diffexpress, level=c('down','ns','up'),order=T)
pdf(paste0('revision/git/figures','/Fig9_volcano_Sim.pdf'), width=2., height=2,pointsize = 10)
ggplot(data = df, aes(x = log2(fc), y = -log10(pval), col = diffexpress)) +
  geom_hline(yintercept = -log10(0.05), col = "gray", linetype = 'dashed') +
  geom_point(size = 1) +
  scale_color_manual(values = c("blue","grey", "orange")) + # to set the labels in case we want to overwrite the categories from the dataframe (UP, DOWN, NO)<br />
  coord_cartesian(ylim = c(0, 3.5), xlim = c(-2, 4)) +
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black"),
        legend.position="none")
dev.off()

####################################################################
####################################################################
#################### supp Figure 5
load(paste0(rootdir,'data/suppFig5_metrics.rda'))

reference=as.matrix(read.csv(paste0(rootdir,"/data/suppFig5_coords_ref_pixel.csv"),row.names=1))
img<-readPNG(paste0(rootdir,"/data/suppFig5_ref_pixel.png"))
h<-dim(img)[1]
w<-dim(img)[2]
pdf(paste0(rootdir,"/figures/suppFig5_ref_pixelWspot.pdf"), width=8, height=8)
par(mar=c(0,0,0,0), xpd=NA, mgp=c(0,0,0), oma=c(0,0,0,0), ann=F)
plot.new()
plot.window(0:1, 0:1)
usr=c(0,1,0,1)
rasterImage(img, usr[1], usr[3], usr[2], usr[4])
points(0,0, cex=1,pch=16, col=rgb(.9,.9,.9,.5))
points(reference[,"x"]/512,(512-reference[,"y"])/512, cex=0.4,pch=16, col=rgb(.2,.2,.2,.5))
dev.off()

for(d in c('low','med','high')){
  ref=suppfig5_coords_ref
  fit=suppfig5_coords_unaligned[[d]]
  maxy=max(max(ref[,2]),max(fit[,2]))
  pdf(paste0(rootdir,"figures/suppFig5_spatial_",d,"_unaligned.pdf"),width=8,height=9.5,pointsize = 10)
  plot(ref[,1],maxy-ref[,2],pch=16,col="grey60",lwd=0.1,xlab="",ylab="")
  points(fit[,1],maxy-fit[,2],pch=4,col='blue',lwd=1.5,xlab="",ylab="")
  dev.off()
}

for(d in c('low','med','high')){
  ref=suppfig5_coords_ref
  fit=suppfig5_coords_stacker[[d]][['t1']]
  maxy=max(max(ref[,2]),max(fit[,2]))
  pdf(paste0(rootdir,"figures/suppFig5_spatial_",d,"_stacker.pdf"),width=8,height=9.5,pointsize = 10)
  plot(ref[,1],maxy-ref[,2],pch=16,col="grey30",lwd=0.1,xlab="",ylab="")
  points(fit[,1],maxy-fit[,2],pch=4,col='red',lwd=1,xlab="",ylab="")
  dev.off()
}

for (d in c('low','med','high')){
  ref=suppfig5_coords_ref
  fit=suppfig5_coords_stu[[d]]
  maxy=max(max(ref[,2]),max(fit[,2]))
  pdf(paste0(rootdir,"figures/suppFig5_spatial_",d,"_STUtility.pdf"),width=8,height=9.5,pointsize = 10)
  plot(ref[,1],maxy-ref[,2],pch=16,col="grey40",lwd=0.1,xlab="",ylab="")
  points(fit[,1],maxy-fit[,2],pch=4,col='red',lwd=1,xlab="",ylab="")
  dev.off()
}

for (d in c('low','med','high')){
  ref=suppfig5_coords_ref
  fit=suppfig5_coords_paste[[d]]
  maxy=max(max(ref[,2]),max(fit[,2]))
  pdf(paste0(rootdir,"figures/suppFig5_spatial_",d,"_PASTE.pdf"),width=8,height=9.5,pointsize = 10)
  plot(ref[,1],maxy-ref[,2],pch=16,col="grey30",lwd=0.1,xlab="",ylab="")
  points(fit[,1],maxy-fit[,2],pch=4,col='red',lwd=1,xlab="",ylab="")
  dev.off()
}

for(d in c('low','med','high')){
  ref=suppfig5_coords_ref
  fit=suppfig5_coords_gpsa[[d]]
  maxy=max(max(ref[,2]),max(fit[,2]))
  pdf(paste0(rootdir,"figures/suppFig5_spatial_",d,"_GPSA.pdf"),width=8,height=9.5,pointsize = 10)
  plot(ref[,1],maxy-ref[,2],pch=16,col="grey30",lwd=0.1,xlab="",ylab="")
  points(fit[,1],maxy-fit[,2],pch=4,col='red',lwd=1,xlab="",ylab="")
  dev.off()
}



df=data.frame(
  "amp"=rep(c('low','med','high'),time=5),
  "method"=c(rep("moving",3),rep('STaCker',3),rep('STUtility',3),rep('PASTE',3),rep("GPSA",3)),
  "MSE"=c(suppfig5_mse_unaligned,c(mean(suppfig5_mse_stacker[['low']]),mean(suppfig5_mse_stacker[['med']]),mean(suppfig5_mse_stacker[['high']])),
          suppfig5_mse_stu,suppfig5_mse_paste,suppfig5_mse_gpsa),
  'SE'=c(rep(0,3),c(sd(suppfig5_mse_stacker[['low']])/sqrt(length(suppfig5_mse_stacker[['low']])),
                    sd(suppfig5_mse_stacker[['med']])/sqrt(length(suppfig5_mse_stacker[['med']])),
                    sd(suppfig5_mse_stacker[['high']])/sqrt(length(suppfig5_mse_stacker[['high']]))),
         rep(0,3),rep(0,3), rep(0,3))
)
df$method=factor(df$method,level=c('moving','STaCker','STUtility','PASTE','GPSA'))

tag='low'
pdf(paste0(rootdir,'figures/suppFig5_MSE_low.pdf'),width=2*5/6,height=1.,pointsize = 20)
ggplot(df[which(tag==df$amp),]) +
  geom_bar( aes(x=method, y=MSE), stat="identity", color="black",fill="white", alpha=0.7,width=0.8) +
  geom_errorbar( aes(x=method, ymin=MSE-SE, ymax=MSE+SE), width=0.1, colour="black", alpha=0.5, size=0.5)+
  ylim(0,0.7)+coord_cartesian(ylim=c(0,0.31))+
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black"),
        axis.title.x=element_blank(),axis.text.x=element_blank(), 
        axis.title.y=element_blank())
dev.off()
tag='low'
pdf(paste0(rootdir,'figures/suppFig5_MSE_low_zoomIn.pdf'),width=2*5/6,height=1.,pointsize = 20)
ggplot(df[which(tag==df$amp),]) +
  geom_bar( aes(x=method, y=MSE), stat="identity", color="black",fill="white", alpha=0.7,width=0.8) +
  geom_errorbar( aes(x=method, ymin=MSE-SE, ymax=MSE+SE), width=0.1, colour="black", alpha=0.5, size=0.5)+
  ylim(0,0.7)+coord_cartesian(ylim=c(0,0.2))+
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black"),
        axis.title.x=element_blank(),axis.text.x=element_blank(), 
        axis.title.y=element_blank())
dev.off()

tag='med'
pdf(paste0(rootdir,'figures/suppFig5_MSE_',tag,'.pdf'),width=2*5/6,height=1.,pointsize = 20)
ggplot(df[which(tag==df$amp),]) +
  geom_bar( aes(x=method, y=MSE), stat="identity", color="black",fill="white", alpha=0.7,width=0.8) +
  geom_errorbar( aes(x=method, ymin=MSE-SE, ymax=MSE+SE), width=0.1, colour="black", alpha=0.5, size=0.5)+
  ylim(0,0.7)+coord_cartesian(ylim=c(0,0.21))+
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black"),
        axis.title.x=element_blank(),axis.text.x=element_blank(), 
        axis.title.y=element_blank())
dev.off()

tag='high'
pdf(paste0(rootdir,'figures/suppFig5_MSE_high.pdf'),width=2*5/6,height=1.2,pointsize = 20)
ggplot(df[which(tag==df$amp),]) +
  geom_bar( aes(x=method, y=MSE), stat="identity", color="black",fill="white", alpha=0.7,width=0.8) +
  geom_errorbar( aes(x=method, ymin=MSE-SE, ymax=MSE+SE), width=0.1, colour="black", alpha=0.5, size=0.5)+
  ylim(0,0.7)+coord_cartesian(ylim=c(0,0.7))+
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black"),
        axis.title.x=element_blank(),axis.text.x=element_blank(), 
        axis.title.y=element_blank())
dev.off()

tag='high'
pdf(paste0(rootdir,'figures/suppFig5_MSE_high_zoomIn.pdf'),width=2*5/6,height=1.2,pointsize = 20)
ggplot(df[which(tag==df$amp),]) +
  geom_bar( aes(x=method, y=MSE), stat="identity", color="black",fill="white", alpha=0.7,width=0.8) +
  geom_errorbar( aes(x=method, ymin=MSE-SE, ymax=MSE+SE), width=0.1, colour="black", alpha=0.5, size=0.5)+
  ylim(0,0.7)+coord_cartesian(ylim=c(0,0.2))+
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black"),
        axis.title.x=element_blank(),axis.text.x=element_blank(), 
        axis.title.y=element_blank())
dev.off()

(t.test(suppfig5_mse_stacker[['low']], mu=suppfig5_mse_unaligned[1]))$p.value
(t.test(suppfig5_mse_stacker[['med']], mu=suppfig5_mse_unaligned[2]))$p.value
(t.test(suppfig5_mse_stacker[['high']], mu=suppfig5_mse_unaligned[3]))$p.value
(t.test(suppfig5_mse_stacker[['low']], mu=suppfig5_mse_stu[1]))$p.value
(t.test(suppfig5_mse_stacker[['med']], mu=suppfig5_mse_stu[2]))$p.value
(t.test(suppfig5_mse_stacker[['high']], mu=suppfig5_mse_stu[3]))$p.value
(t.test(suppfig5_mse_stacker[['low']], mu=suppfig5_mse_paste[1]))$p.value
(t.test(suppfig5_mse_stacker[['med']], mu=suppfig5_mse_paste[2]))$p.value
(t.test(suppfig5_mse_stacker[['high']], mu=suppfig5_mse_paste[3]))$p.value
(t.test(suppfig5_mse_stacker[['low']], mu=suppfig5_mse_gpsa[1]))$p.value
(t.test(suppfig5_mse_stacker[['med']], mu=suppfig5_mse_gpsa[2]))$p.value
(t.test(suppfig5_mse_stacker[['high']], mu=suppfig5_mse_gpsa[3]))$p.value

