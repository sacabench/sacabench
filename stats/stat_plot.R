#/*******************************************************************************
# * Copyright (C) 2018 Rosa Pink <rosa.pink@tu-dortmund.de>
# * Copyright (C) 2018 Hermann Foot <hermann.foot@tu-dortmund.de>
# * Copyright (C) 2018 Johannes Bahne <johannes.bahne@tu-dortmund.de>
# *
# * All rights reserved. Published under the BSD-3 license in the LICENSE file.
# ******************************************************************************/
 
library("rjson")
library("RColorBrewer")

args <- commandArgs(TRUE)

print(args)


extract_stats <-function(json_name)
{
  data= fromJSON(file=json_name,simplify = TRUE)
  name=data$stats[1][[1]]$value
  algorithm = data$sub[1][[1]]$sub[[4]]
  runtime=algorithm$timeEnd-algorithm$timeStart
  mem= algorithm$memPeak
  
  return(list(name=name,time=runtime,memory=mem))
}

extract_details<-function(json_name)
{
  data= fromJSON(file=json_name,simplify = TRUE)[[1]]
  print(data)
  #Algorithm
  algorithm= data$sub[1][[1]]$sub[[4]]
  #Phases
  phases= algorithm$sub
  
  
  #get main entries
  algorithm_name = data$stats[[1]]$value
  max_mem = algorithm$memPeak/1000
  runtime_overall = (algorithm$timeEnd - algorithm$timeStart)/1000

  
  #get entries for every phase
  if(length(phases)!=0){
    
    phase_names = phases[[1]]$title
    phase_runtimes = (phases[[1]]$timeEnd-phases[[1]]$timeStart)/1000
    phase_mems  = phases[[1]]$memPeak/1000

    
    for(i in 2:length(phases)){
      phase_names = cbind(phase_names, phases[[i]]$title)
      phase_runtimes = rbind(phase_runtimes, 
                             (phases[[i]]$timeEnd-phases[[i]]$timeStart)/1000)
      phase_mems  = rbind(phase_mems, phases[[i]]$memPeak/1000)
    }
    
    phase_names = cbind("overall", phase_names)
    phase_runtimes = rbind(runtime_overall, phase_runtimes)
    phase_mems = rbind(max_mem, phase_mems)
  }else{
    phase_names = "overall"
    phase_runtimes = (data$timeEnd-data$timeStart)/1000
    phase_mems = data$memPeak/1000
  }
  label_runtime = "in seconds"
  label_mem = "in KB"
  
  #If values are too big -> next unit
  too_long = min(phase_runtimes) > 60
  too_big  = min(phase_mems) > 10000 
  
  if(too_long){
    phase_runtimes = phase_runtimes/60
    label_runtime = "in minutes"
  }
  if(too_big){
    phase_mems = phase_mems / 1000
    label_mem = "in MB"
  }
  
  print(phase_runtimes)
  print(phase_mems)
  #plot
  plot_benchmark_single(algorithm_name, phase_names, 
                        phase_runtimes, phase_mems, label_runtime, label_mem)
}

plot_benchmark_single<-function(algorithm_name, phase_names, runtimes, mems,
                                label_runtime, label_mem)
{
  par(mfrow=c(1,2),mai=c(0.7,1,1,1))
  
  #Plots for runtimes in each phase
  if(min(runtimes) < 1){
    barplot(runtimes[1],beside=FALSE,col = 2, ylab = label_runtime,  yaxt="n")
    barplot(as.matrix(runtimes[2:length(runtimes)]),beside=FALSE,
            col = 3:(length(runtimes)+2), add = TRUE)
    title("Runtime", line=1)
  }else{
    barplot(runtimes[1],beside=FALSE,col = 2, ylab = label_runtime,  yaxt="n")
    barplot(as.matrix(runtimes[2:length(runtimes)]),beside=FALSE,
            col = 3:(length(runtimes)+2), add = TRUE, yaxt="n")
    axis(2,at=seq(0,max(runtimes), by = ceiling(max(runtimes)/10)),
         labels=format(seq(0,max(runtimes),by = ceiling(max(runtimes)/10)),scientific=FALSE))
    title("Runtime", line=1)
  }
  
  
  #Plots for peak memory usage in each phase
  barplot(mems, beside=TRUE, col = 2:(length(mems)+1), ylab = label_mem,  yaxt="n")
  axis(2,at=seq(0,max(mems), by = round(max(mems)/10, digits = 0)),
       labels=format(seq(0,max(mems),by = round(max(mems)/10, digits = 0)),scientific=FALSE))
  title("Memory peak", line=1)
  
  #Header and Footer
  mtext(algorithm_name, side=3, outer=TRUE, line=-2,cex = 2)
  par(fig = c(0, 1, 0, 1), oma = c(0, 0, 0, 0), mar = c(0, 0, 0, 0), 
      mai=c(1,1,1,1), new = TRUE)
  legend("bottom", legend = phase_names, inset = c(0,-0.1), 
         col = 2:(length(mems)+1), pch = 15, xpd = TRUE, horiz = TRUE)
}

plot_benchmark_multi_scatter<-function(algorithm_names, runtimes, mems, logarithmic, 
                               label_runtime, label_mem, label_main, cols)
{
  par(mfrow=c(1,1),mai=c(1,1,1,2), oma = c(0,0,0,2))
  plot(runtimes, mems, col = cols, pch = 19, 
       xlab = label_runtime, 
       ylab = label_mem, 
       main = label_main, log = logarithmic, xaxt = "n", yaxt="n")
  axis(1,at=seq(0,max(runtimes), by = max(runtimes)/20),
       labels=format(seq(0,max(runtimes), 
                         by = max(runtimes)/20),scientific=FALSE))
  axis(2,at=seq(0,max(mems), by = max(mems)/10),
       labels=format(seq(0,max(mems), by = max(mems)/10),scientific=FALSE))
  #text(runtimes, mems, labels=algorithm_names, 
  #     col = 2:(length(runtimes)+1), adj=c(0.3,-0.35))
  par(fig = c(0, 1, 0, 1), oma = c(0, 0, 0, 0), mar = c(0, 0, 0, 0), 
      mai=c(1,1,1,1), new = TRUE)
  legend("right",  legend = algorithm_names, col = cols, 
         inset = c(-0.3,0), pch = 19, xpd = TRUE, bty = "n")
}

plot_benchmark_multi_bar<-function(algorithm_names, runtimes, mems, 
                                 label_runtime, label_mem, label_main, cols)
{
  par(mfrow=c(2,1),mai=c(1, 1, 0.3, 0))
  barplot(runtimes,beside=TRUE,col = cols,
          ylab = label_runtime, yaxt="n", names.arg = algorithm_names,
          main = "Memory & runtime measurements", las = 2)
  axis(2,at=seq(0,max(runtimes), by = max(runtimes)/10),
       labels=format(seq(0,max(runtimes), by = max(runtimes)/10),
                     scientific=FALSE))
  
  
  par(mai=c(0.3, 1, 0.3, 0))
  barplot(mems,beside=TRUE, col = cols,
          add = FALSE, ylab = label_mem, ylim = c(max(mems),0),  yaxt="n")
  axis(2,at=seq(0,max(mems), by = max(mems)/10),
       labels=format(seq(0,max(mems),by = max(mems)/10),scientific=FALSE))
  
  #par(fig = c(0, 1, 0, 1), oma = c(0, 0, 0, 0), mar = c(0, 0, 0, 0), 
  #    mai=c(1,1,1,1), new = TRUE)
  #legend("right",  legend = algorithm_names, col = 2:(length(runtimes)+1), 
  #       inset = c(-0.125,0), pch = 19, xpd = TRUE, bty = "n")
}

pareto_optimal = function(x,y){
  dif = x-y
  better= sum(any(dif<0))
  
  #there is no better one
  if(better==0) 
  {
    return(FALSE)
  }
  #at least one better (dominating or incomparable)
  return(TRUE)
}

calculate_paretofront<-function(x)
{
  n=dim(x)[1]
  result =rep(FALSE, n)
  for(index in 1:n)
  {
    for(other_index in 1:n)
    {
      if(other_index!=index)
      {
        if(!pareto_optimal(x[index,],x[other_index,]))
        {
          break
        }
      }
      if(other_index==n)
      {
        result[index]=TRUE
      }
    }
  }
  return(result)
}

example_plot_multi <-function(){
  names = c("DC3", "DC7", "BPR", "nzSufSort", "mSufSort", "DivSufSort",
            "GSAKA", "Deep-Shallow", "SAKA-K", "SADS", "SAIS", "gsa-is",
            "SAIS-light", "qSufSort")
  runtimes = c(200,300,200,500,50,100,15,10, 1, 1000,250,100,300,600)
  
  mems = c(75000,10000,8000,10000,9400,50000,9500, 10000, 100000,
           150000,12000,15000,90000,120000)
  
  pareto = F
  logarithmic = F
  
  label_runtime = "Runtime in seconds"
  label_mem = "Memory peak in KB"
  label_main = "Memory & runtime measurements"
  
  n <- length(names)
  plot_benchmark_multi_bar(names, runtimes, mems, label_runtime,
                         label_mem, label_main, getDistinctColors(n))
  
  if(logarithmic){
    label_main = paste(label_main, "(logarithmic scale)", sep = " ")
    logarithmic = "xy"
  }else{
    logarithmic = ""
  }
  
  if(pareto){
    algo_data = cbind(mems, runtimes)
    
    is_pareto = calculate_paretofront(algo_data)
    pareto_inidices = which(is_pareto)
    
    names = names[pareto_inidices]
    runtimes = runtimes[pareto_inidices]
    mems = mems[pareto_inidices]
    
    label_main = paste(label_main, "- Paretofront", sep = " ")
  }
  
  #If values are too big -> next unit
  too_long = min(runtimes) > 60
  too_big  = min(mems) > 10000 
  if(too_long){
    runtimes = runtimes/60
    label_runtime = "Runtime in minutes"
  }
  if(too_big){
    mems = mems / 1000
    label_mem = "Memory peak in MB"
  }
  n <- length(names)
  plot_benchmark_multi_scatter(names, runtimes, mems, logarithmic,
                       label_runtime, label_mem,
                       label_main, getDistinctColors(n))
}

getDistinctColors <- function(n) {
  qual_col_pals <- brewer.pal.info[brewer.pal.info$category == 'qual',]
  col_vector <- unique (unlist(mapply(brewer.pal,
                                      qual_col_pals$maxcolors,
                                      rownames(qual_col_pals))));
  stopifnot (n <= length(col_vector));
  xxx <- col2rgb(col_vector);
  dist_mat <- as.matrix(dist(t(xxx)));
  diag(dist_mat) <- 1e10;
  while (length(col_vector) > n) {
    minv <- apply (dist_mat,1,function(x)min(x));
    idx <- which(minv==min(minv))[1];
    dist_mat <- dist_mat[-idx, -idx];
    col_vector <- col_vector[-idx]
  }
  return(col_vector)
}
#testList= list(name="a",x=1,y=2)
#testList2= list(name="b",x=2,y=1)

#datafra=data.frame(name=c("d","e"),x=c(3,2),y=c(1,5),
#                   stringsAsFactors = FALSE)
print(args)
extract_details(args)
example_plot_multi()

#for command line:
#R -e 'install.packages("package", repos="http://cran.us.r-project.org")'

#myscript.r --file myfile.txt

#run following command to execute the script:
#R CMD BATCH stat_plot.R 