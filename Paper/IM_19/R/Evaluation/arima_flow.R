
setEPS()
postscript("/home/anle/Dropbox/Research/TM_estimation_RNN/Paper/IM_19/R/Evaluation/arima_flow_142_day_10.eps",height=2.5, width=4.5)
par(mar=c(4, 4, 0.5, 0.0) + 0.1)
par(mgp = c(3, 1, 0))

data <- read.table("/home/anle/Dropbox/Research/TM_estimation_RNN/Paper/IM_19/R/Evaluation/arima_flow_142_day_10.csv",header=T,sep=",")

y_max = max(data$ACTUAL, data$PREDICTION)
x_max = max(data$N)
y_min = min(data$ACTUAL, data$PREDICTION)
x_min = min(data$N)
print(y_max)

measured_points <- read.table("/home/anle/Dropbox/Research/TM_estimation_RNN/Paper/IM_19/R/Evaluation/arima_flow_142_day_10_measured_points.csv",header=T,sep=",")

# dataError <- read.table("all_max_stretch_error_multi_topo_e20n6.dat",header=T,sep="\t")
# errorY0 = c(dataError$trueCircle21, dataError$trueRecrangle31, dataError$random62, dataError$random15, dataError$random26)
# errorMargin=c(dataError$trueCircle21Xi, dataError$trueRecrangle31Xi, dataError$random62Xi, dataError$random15Xi, dataError$random26Xi)
plot_colors <- c("black", "red")



plot(c(x_min, x_max + 1),c(y_min*1000,y_max*1000 + 2), type="n", cex.lab = 1.1, cex.axis=1.5, xlab="Timestep", ylab="Traffic volume (Mbps)")
lines(data$N, data$ACTUAL*1000, lty=1, col=plot_colors[1], lwd=1)
lines(data$N, data$PREDICTION*1000, lty=1, col=plot_colors[2], lwd=1)

legend("topright",c("Actual Traffic", "Predicted Traffic", "Measured Points"), lty=c(1,1, NA), pch=c(NA, NA, '*'), col=c("black", "red", "darkblue"), cex=1, xjust=0.5, yjust=0.5, ncol=1)


#legend("topright", bty = "n", legend=c("ARIMA_1","ARIMA_2","ARIMA_4", "ARIMA_5","BiRNN"), 
#       ncol=2, fill=plot_colors, cex=1, col=plot_colors, angle=c(100,35,120,135,90),density=c(6,12,20,10,100))

points(measured_points$X, measured_points$Y*1000, pch="*", col="darkblue", cex=2)

dev.off()



