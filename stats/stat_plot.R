#/*******************************************************************************
# * Copyright (C) 2018 Rosa Pink <rosa.pink@tu-dortmund.de>
# * Copyright (C) 2018 Hermann Foot <hermann.foot@tu-dortmund.de>
# * Copyright (C) 2018 Johannes Bahne <johannes.bahne@tu-dortmund.de>
# *
# * All rights reserved. Published under the BSD-3 license in the LICENSE file.
# ******************************************************************************/
 
library("rjson")

extract_stats <-function(json_name)
{
  data= fromJSON(file=json_name,simplify = TRUE)
  name=data$stats[1][[1]]$value
  algorithm = data$sub[1][[1]]$sub[[4]]
  runtime=algorithm$timeEnd-algorithm$timeStart
  mem= algorithm$memPeak
  
  return(list(name=name,time=runtime,memory=mem))
}

plot_benchmark_single<-function(algorithm_name, phase_names, runtimes, mems)
{
  par(mfrow=c(1,2),mai=c(0.7,1,1,1))
  
  #Plots for runtimes in each phase
  barplot(runtimes[1],beside=FALSE,col = 2, ylab = "in seconds")
  barplot(as.matrix(runtimes[2:length(runtimes)]),beside=FALSE,
          col = 3:(length(runtimes)+2), add = TRUE)
  title("Runtime", line=1)
  
  #Plots for peak memory usage in each phase
  barplot(mems, beside=TRUE, col = 2:(length(mems)+1), ylab = "in KB")
  title("Memory Usage", line=1)
  
  #Header and Footer
  mtext(algorithm_name, side=3, outer=TRUE, line=-2,cex = 2)
  par(fig = c(0, 1, 0, 1), oma = c(0, 0, 0, 0), mar = c(0, 0, 0, 0), 
      mai=c(1,1,1,1), new = TRUE)
  legend("bottom", legend = phase_names, inset = c(0,-0.1), 
         col = 2:(length(mems)+1), pch = 15, xpd = TRUE, horiz = TRUE)
}

extract_details<-function(json_name)
{
  data= fromJSON(file=json_name,simplify = TRUE)
  #Algorithm
  algorithm= data$sub[1][[1]]$sub[[4]]
  #Phases
  phases= algorithm$sub
  
  #get main entries
  algorithm_name = data$stats[[1]]$value
  max_mem = algorithm$memPeak
  runtime_overall = algorithm$timeEnd - algorithm$timeStart
  
  #get entries for every phase
  phase_names = phases[[1]]$title
  phase_runtimes = phases[[1]]$timeEnd-phases[[1]]$timeStart
  phase_mems  = phases[[1]]$memPeak
  
  for(i in 2:length(phases)){
    phase_names = cbind(phase_names, phases[[i]]$title)
    phase_runtimes = rbind(phase_runtimes, 
                           phases[[i]]$timeEnd-phases[[i]]$timeStart)
    phase_mems  = rbind(phase_mems, phases[[i]]$memPeak)
  }
  
  phase_names = cbind("overall", phase_names)
  phase_runtimes = rbind(runtime_overall, phase_runtimes)
  phase_mems = rbind(max_mem, phase_mems)
  
  #plot
  plot_benchmark_single(algorithm_name, phase_names, 
                        phase_runtimes, phase_mems)
}

#testList= list(name="a",x=1,y=2)
#testList2= list(name="b",x=2,y=1)

#datafra=data.frame(name=c("d","e"),x=c(3,2),y=c(1,5),
#                   stringsAsFactors = FALSE)
