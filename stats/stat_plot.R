#/*******************************************************************************
# * Copyright (C) 2018 Rosa Pink <rosa.pink@tu-dortmund.de>
# * Copyright (C) 2018 Hermann Foot <hermann.foot@tu-dortmund.de>
# * Copyright (C) 2018 Johannes Bahne <johannes.bahne@tu-dortmund.de>
# *
# * All rights reserved. Published under the BSD-3 license in the LICENSE file.
# ******************************************************************************/
 

install.packages("rjson")
install.packages("RColorBrewer")
library("rjson")
library("RColorBrewer")

args <- commandArgs(TRUE)

extract_details<-function(json_name)
{
  data = fromJSON(file=json_name,simplify = TRUE)
  algorithm_name = data[[1]]$stats[[1]]$value

  for(run in 1:length(data)){
    #Algorithm
    algorithm = data[[run]]$sub[1][[1]]$sub[[4]]
    
    #Phases
    phases = algorithm$sub

    #get main entries
    max_mem = algorithm$memPeak/1000
    runtime_overall = (algorithm$timeEnd - algorithm$timeStart)
    
    #get entries for every phase
    if(length(phases)!=0){
      
      phase_names = phases[[1]]$title
      tmp_phase_runtimes = (phases[[1]]$timeEnd-phases[[1]]$timeStart)
      tmp_phase_mems  = phases[[1]]$memPeak/1000
      
      
      for(i in 2:length(phases)){
        phase_names = cbind(phase_names, phases[[i]]$title)
        tmp_phase_runtimes = rbind(tmp_phase_runtimes, 
                               (phases[[i]]$timeEnd-phases[[i]]$timeStart))
        tmp_phase_mems  = rbind(tmp_phase_mems, phases[[i]]$memPeak/1000)
      }
      
      phase_names = cbind("overall", phase_names)
      tmp_phase_runtimes = rbind(runtime_overall[1], tmp_phase_runtimes)
      tmp_phase_mems = rbind(max_mem, tmp_phase_mems)
    }else{
      phase_names = "overall"
      
      tmp_phase_runtimes = (algorithm$timeEnd-algorithm$timeStart)
      tmp_phase_mems = algorithm$memPeak/1000
    }
    if(run == 1){
      phase_runtimes = tmp_phase_runtimes
      phase_mems = tmp_phase_mems
    }else{
      phase_runtimes = cbind(phase_runtimes, tmp_phase_runtimes)
      phase_mems = cbind(phase_mems, tmp_phase_mems)
    }
  }
  
  if(length(phases)!=0){
    phase_runtimes = rowMeans(phase_runtimes)
    phase_mems = rowMeans(phase_mems)
  }
  
  label_runtime = "in milliseconds"
  label_mem = "in KB"
  
  #If values are too big -> next unit
  too_big  = min(phase_mems) > 10000 
  if(min(phase_runtimes) > 100){
    phase_runtimes = phase_runtimes/1000
    label_runtime = "Runtime in seconds"
  }else if(min(phase_runtimes)>1000*60){
    phase_runtimes = phase_runtimes/(1000*60)
    label_runtime = "Runtime in minutes"
  }
  if(too_big){
    phase_mems = phase_mems / 1000
    label_mem = "Memory peak in MB"
  }
  
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
  header_name = paste(algorithm_name," (",args[3])
  if(!is.na(args[4])){
    header_name = paste(header_name, ", size:", args[4],")")
  }else{
    header_name = paste(header_name,")");
  }
  mtext(header_name, side=3, outer=TRUE, line=-2,cex = 2)
  par(fig = c(0, 1, 0, 1), oma = c(0, 0, 0, 0), mar = c(0, 0, 0, 0), 
      mai=c(1,1,1,1), new = TRUE)
  legend("bottom", legend = phase_names, inset = c(0,-0.1), 
         col = 2:(length(mems)+1), pch = 15, xpd = TRUE, horiz = TRUE)
}

plot_benchmark_multi_scatter<-function(algorithm_names, runtimes, mems, logarithmic, 
                               label_runtime, label_mem, label_main, cols)
{
  par(mfrow=c(1,1),mai=c(1,1,1,1), oma = c(0,0,0,2))
  plot(runtimes, mems, col = cols, pch = 19, 
       xlab = label_runtime, 
       ylab = label_mem, 
       main = label_main, log = logarithmic, xaxt = "n", yaxt="n")
  axis(1,at=seq(0,max(runtimes), by = ceiling(max(runtimes)/20)),
       labels=format(seq(0,max(runtimes), 
                         by = ceiling(max(runtimes)/20)),scientific=FALSE))
  axis(2,at=seq(0,max(mems), by = ceiling(max(mems)/10)),
       labels=format(seq(0,max(mems), by = ceiling(max(mems)/10)),scientific=FALSE))
  text(runtimes, mems, labels=algorithm_names, xpd = TRUE,
       col = cols, adj=c(0.3,-0.35))
  par(fig = c(0, 1, 0, 1), oma = c(0, 0, 0, 0), mar = c(0, 0, 0, 0), 
      mai=c(1,1,1,1), new = TRUE)
  legend("right",  legend = algorithm_names, col = cols, 
         inset = c(-0.23,0), pch = 19, xpd = TRUE, bty = "n")
}

plot_benchmark_multi_bar<-function(algorithm_names, runtimes, mems, 
                                 label_runtime, label_mem, label_main, cols)
{
  par(mfrow=c(2,1),mai=c(1, 1, 0.3, 0))
  barplot(runtimes,beside=TRUE,col = cols,
          ylab = label_runtime, yaxt="n", names.arg = algorithm_names,
          main = label_main, las = 2)
  axis(2,at=seq(0,max(runtimes), by = ceiling(max(runtimes)/10)),
       labels=format(seq(0,max(runtimes), by = ceiling(max(runtimes)/10)),
                     scientific=FALSE))
  
  par(mai=c(0.3, 1, 0.3, 0))
  barplot(matrix(mems),beside=TRUE, col = cols,
          ylab = label_mem, yaxt="n", ylim = c(max(mems),0))
  axis(2,at=seq(0,max(mems), by = ceiling(max(mems)/10)),
       labels=format(seq(0,max(mems),by = ceiling(max(mems)/10)),scientific=FALSE))
  
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
  prepare_plot_data(names, runtimes, mems, pareto, logarithmic)
  
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

extract_all_stats <- function(json_name, pareto = F, logarithmic = F){
  data= fromJSON(file=json_name,simplify = TRUE)
  
  #names
  algorithm_names = data[[1]][[1]]$stats[[1]]$value
  for(no_algos in 1:length(data)){
    if(no_algos != 1){
      algorithm_names = cbind(algorithm_names, data[[no_algos]][[1]]$stats[[1]]$value)
    }
    #Algorithm
    algorithm = data[[no_algos]][[1]]$sub[[1]]$sub[[4]]
    
    #runtime
    tmp_runtimes = (algorithm$timeEnd - algorithm$timeStart)
    #mem
    tmp_mems = algorithm$memPeak/1000
    
    if(length(data[[no_algos]]) >= 2){
      for(runs in 2:length(data[[no_algos]])){
        #Algorithm
        algorithm = data[[no_algos]][[runs]]$sub[[1]]$sub[[4]]
        
        #runtime
        tmp_runtimes = cbind(tmp_runtimes,(algorithm$timeEnd - algorithm$timeStart))
        
        #mem
        tmp_mems = cbind(tmp_mems,algorithm$memPeak/1000)
      }
    }
    
    if(no_algos == 1){
      runtimes = mean(tmp_runtimes)
      mems = mean(tmp_mems)
    }else{
      runtimes = cbind(runtimes, mean(tmp_runtimes))
      mems = cbind(mems, mean(tmp_mems))
    }
  }
  
  prepare_plot_data(algorithm_names[1,], runtimes[1,], mems[1,], pareto, logarithmic)
}

prepare_plot_data <- function(names, runtimes, mems, pareto, logarithmic){
  
  label_runtime = "Runtime in milliseconds"
  label_mem = "Memory peak in KB"
  label_main = "Memory & runtime measurements"
  header_name = paste(label_main," (",args[3])
  if(!is.na(args[4])){
    header_name = paste(header_name, ", size:", args[4],")")
  }else{
    header_name = paste(header_name,")");
  }
  
  n <- length(names)
  plot_benchmark_multi_bar(names, runtimes, mems, label_runtime,
                           label_mem, header_name, getDistinctColors(n))
  
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
  
  label_main = paste(label_main," (",args[3])
  if(!is.na(args[4])){
    label_main = paste(label_main, ", size:", args[4],")")
  }else{
    label_main = paste(label_main,")");
  }
  
  #If values are too big -> next unit
  too_big  = min(mems) > 10000 
  if(min(runtimes) > 100){
    runtimes = runtimes/1000
    label_runtime = "Runtime in seconds"
  }else if(min(runtimes)>1000*60){
    runtimes = runtimes/(1000*60)
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

#testList= list(name="a",x=1,y=2)
#testList2= list(name="b",x=2,y=1)

#datafra=data.frame(name=c("d","e"),x=c(3,2),y=c(1,5),
#                   stringsAsFactors = FALSE)

if(args[2] == 0){
  extract_details(args[1])
}else{
  extract_all_stats(args[1])
  extract_all_stats(args[1],pareto = T)
}

#for command line:
#R -e 'install.packages("package", repos="http://cran.us.r-project.org")'
