setwd("C:/Users/Hao/Documents/Fall2016/COMP551/Project3")
f = file("train_x.bin", "rb")
x = readBin(f,'integer',n=100000*60*60,size=1,signed=F)
close(f)
x = matrix(x, ncol=60*60, byrow=T) # each row is an image
matrix<-matrix(x[14,], ncol=60, byrow=T)
image(matrix, col=gray(12:1/12))
library(tmap)
bb(matrix)




even<-seq(2,3600,2)
zz<-x[,even]

image(matrix(zz[13,], ncol=30, byrow=T), col=gray(12:1/12))
y=read.csv("train_y.csv")
z<-data.frame(cbind(zz,y$Prediction))

xnam<-paste("X",1:1800,sep = "")
fmla<-as.formula(paste("X1801~",paste(xnam,collapse = "+")))

fit3<-glm(fmla,family =poisson, z[3000:6000,])
fit3$deviance/fit3$df.null

fit4<-glm(fmla,family =poisson, z[1:3500,])
fit4$deviance/fit4$df.null

zz<-data.frame(zz)
predict<-round(predict(fit3,zz[1:10000,],type="response"),0)

yy<-cbind(predict,y[1:10000,])
yy$p<-yy$predict-yy$Prediction
table(yy$p)
