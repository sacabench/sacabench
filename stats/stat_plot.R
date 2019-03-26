#/*******************************************************************************
# * Copyright (C) 2018 Rosa Pink <rosa.pink@tu-dortmund.de>
# * Copyright (C) 2018 Hermann Foot <hermann.foot@tu-dortmund.de>
# * Copyright (C) 2018 Johannes Bahne <johannes.bahne@tu-dortmund.de>
# *
# * All rights reserved. Published under the BSD-3 license in the LICENSE file.
# ******************************************************************************/

chooseCRANmirror(ind=1)

dir.create("./.rlib")
install.packages("rjson", lib="./.rlib")
install.packages("RColorBrewer", lib="./.rlib")
library("rjson", lib="./.rlib")
library("RColorBrewer", lib="./.rlib")

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
    max_mem = algorithm$memPeak/1024
    runtime_overall = (algorithm$timeEnd - algorithm$timeStart)
    
    #get entries for every phase
    if(length(phases)!=0){
      
      phase_names = phases[[1]]$title
      tmp_phase_runtimes = (phases[[1]]$timeEnd-phases[[1]]$timeStart)
      tmp_phase_mems  = phases[[1]]$memPeak/1024
      
      
      for(i in 2:length(phases)){
        phase_names = cbind(phase_names, phases[[i]]$title)
        tmp_phase_runtimes = rbind(tmp_phase_runtimes, 
                               (phases[[i]]$timeEnd-phases[[i]]$timeStart))
        tmp_phase_mems  = rbind(tmp_phase_mems, phases[[i]]$memPeak/1024)
      }
      
      phase_names = cbind("overall", phase_names)
      tmp_phase_runtimes = rbind(runtime_overall[1], tmp_phase_runtimes)
      tmp_phase_mems = rbind(max_mem, tmp_phase_mems)
    }else{
      phase_names = "overall"
      
      tmp_phase_runtimes = (algorithm$timeEnd-algorithm$timeStart)
      tmp_phase_mems = algorithm$memPeak/1024
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
  label_mem = "in KiB"
  
  #If values are too big -> next unit
  if(max(phase_runtimes)>1000*60){
    phase_runtimes = phase_runtimes/(1000*60)
    label_runtime = "Runtime in minutes"
  }else if(max(phase_runtimes) > 100){
    phase_runtimes = phase_runtimes/1000
    label_runtime = "Runtime in seconds"
  }
  if(max(phase_mems) > 10*1024*1024){
    phase_mems = phase_mems / 1024 / 1024
    label_mem = "Additional memory peak in GiB"
  }else if(max(phase_mems) > 10 * 1024){
    phase_mems = phase_mems / 1024
    label_mem = "Additional memory peak in MiB"
  }
  
  #plot
  plot_benchmark_single(algorithm_name, phase_names, 
                        phase_runtimes, phase_mems, label_runtime, label_mem)
}

plot_benchmark_single<-function(algorithm_name, phase_names, runtimes, mems,
                                label_runtime, label_mem)
{
  n <- length(phase_names)
  cols = getDistinctColors(n)
  par(mfrow=c(1,2),mai=c(1,1,1,1))
  
  #Plots for runtimes in each phase
  if(min(runtimes) < 1){
    barplot(runtimes[1],beside=FALSE,col = cols[1], ylab = label_runtime,  yaxt="n")
    barplot(as.matrix(runtimes[2:length(runtimes)]),beside=FALSE,
            col = cols[2:n], add = TRUE, ylim=range(pretty(c(0, runtimes))))
    title("Runtime", line=1)
  }else{
    barplot(runtimes[1],beside=FALSE,col = cols[1], ylab = label_runtime,  yaxt="n")
    barplot(as.matrix(runtimes[2:length(runtimes)]),beside=FALSE,
            col = cols[2:n], add = TRUE, yaxt="n")
    axis(2,at=seq(0,max(runtimes), by = ceiling(max(runtimes)/10)),
         labels=format(seq(0,max(runtimes),by = ceiling(max(runtimes)/10)),scientific=FALSE))
    title("Runtime", line=1)
  }
  
  
  #Plots for peak memory usage in each phase
  barplot(matrix(mems), beside=TRUE, col = cols, ylab = label_mem,  yaxt="n")
  axis(2,at=seq(0,max(mems), by = round(max(mems)/10, digits = 0)),
       labels=format(seq(0,max(mems),by = round(max(mems)/10, digits = 0)),scientific=FALSE))
  title("Additional memory peak", line=1)
  
  #Header and Footer
  header_name = paste(algorithm_name," (",args[3], sep = "")
  size = as.numeric(args[4])
  label_unit = "Bytes"
  if(size >= 1024 * 1024 * 1024){
    size = round(size / 1024 / 1024 / 1024, digits = 2)
    label_unit = "GiB"
  }else if(size >= 1024 * 1024){
    size = round(size / 1024 / 1024, digits = 2)
    label_unit = "MiB"
  }else if(size >= 1024){
    size = round(size / 1024, digits = 2)
    label_unit = "KiB"
  } 
  
    header_name = paste(header_name, ", size: ",size,label_unit, ")", sep = "")
  
  mtext(header_name, side=3, outer=TRUE, line=-2,cex = 2)
  par(fig = c(0, 1, 0, 1), oma = c(0, 0, 0, 0), mar = c(0, 0, 0, 0), 
      mai=c(1,1,1,1), new = TRUE)
  if(length(phase_names) > 16){
    op <- par(cex = 0.2)
  }else if(length(phase_names) > 12){
    op <- par(cex = 0.25)
  }else if(length(phase_names) > 6){
    op <- par(cex = 0.5)
  }
  
  legend("bottom", legend = phase_names, inset = c(-10,-0.15), 
         col = cols, pch = 15, xpd = TRUE, ncol = ceiling(length(phase_names)/2), bty = "n")
  op <- par(cex = 1.0)
}

plot_benchmark_multi_scatter<-function(algorithm_names, runtimes, mems, logarithmic, 
                                       label_runtime, label_mem, label_main, cols)
{
  par(mfrow=c(1,1),mai=c(1,1,0.5,2), oma = c(0,0,0,2))
  plot(runtimes, mems, col = cols, pch = 19, 
       xlab = label_runtime, 
       ylab = label_mem, 
       log = logarithmic, xaxt = "n", yaxt="n")
  
  mtext(label_main, side=3, outer=TRUE, line=-2,cex = 1)
  
  if(max(runtimes)<1){
    axis(1,at=seq(0,max(runtimes), by = max(runtimes)/10),
         labels=format(seq(0,max(runtimes), 
                           by = max(runtimes)/10),scientific=FALSE))
  }else{
    axis(1,at=seq(0,max(runtimes), by = ceiling(max(runtimes)/20)),
         labels=format(seq(0,max(runtimes), 
                           by = ceiling(max(runtimes)/20)),scientific=FALSE))
  }
  
  axis(2,at=seq(0,max(mems), by = ceiling(max(mems)/10)),
       labels=format(seq(0,max(mems), by = ceiling(max(mems)/10)),scientific=FALSE))
  
  abline(h=seq(0,max(mems), by = ceiling(max(mems)/10)), v=seq(0,max(runtimes),
                                                               by = ceiling(max(runtimes)/10)),
                                                               lty=6, col = "gray60")
  #text(runtimes, mems, labels=algorithm_names, xpd = TRUE,
  #     col = cols, adj=c(0.3,-0.35))
  #par(fig = c(0, 1, 0, 1), oma = c(0, 0, 0, 0), mar = c(0, 0, 0, 0), 
  #    mai=c(1,1,1,1), new = TRUE)
  legend("right",  legend = algorithm_names, col = cols, 
         inset = c(-0.4,0), pch = 19, xpd = TRUE, bty = "n")
}

plot_benchmark_multi_bar<-function(algorithm_names, runtimes, mems, 
                                 label_runtime, label_mem, label_main, cols)
{
  par(mfrow=c(2,1),mai=c(1, 1, 0.5, 0))
  barplot(runtimes,beside=TRUE,col = cols,
          ylab = label_runtime, yaxt="n", names.arg = algorithm_names, las = 2, space = 0)
  mtext(label_main, side=3, outer=TRUE, line=-2,cex = 1)
  axis(2,at=seq(0,max(runtimes)+max(runtimes)/10, by = ceiling(max(runtimes)/10)),
       labels=format(seq(0,max(runtimes)+max(runtimes)/10, by = ceiling(max(runtimes)/10)),
                     scientific=FALSE))

  abline(h=seq(0,max(runtimes), by = ceiling(max(runtimes)/10)), lty=6, col = "gray60")
  par(mai=c(0.3, 1, 0.5, 0))
  barplot(matrix(mems),beside=TRUE, col = cols,
          ylab = label_mem, yaxt="n", ylim = c(max(mems),0))
  axis(2,at=seq(0,max(mems)+max(mems)/10, by = ceiling(max(mems)/10)),
       labels=format(seq(0,max(mems)+max(mems)/10,by = ceiling(max(mems)/10)),scientific=FALSE))

  abline(h=seq(0,max(mems), by = ceiling(max(mems)/10)), lty=6, col = "gray60")
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

extract_all_stats <- function(json_name, pareto = F, logarithmic = F, plot_scatter = F, sort_by = "none", memory_additional = T){
  data= fromJSON(file=json_name,simplify = TRUE)
  
  #names
  algorithm_names = data[[1]][[1]]$stats[[1]]$value
  for(no_algos in 1:length(data)){
    if(no_algos != 1){
      algorithm_names = cbind(algorithm_names, data[[no_algos]][[1]]$stats[[1]]$value)
    }
    #Algorithm
    if(memory_additional == T){
      algorithm = data[[no_algos]][[1]]$sub[[1]]$sub[[4]]
    }else{
      algorithm = data[[no_algos]][[1]]$sub[[1]]
    }
    
    #runtime
    tmp_runtimes = (algorithm$timeEnd - algorithm$timeStart)
    #mem
    tmp_mems = algorithm$memPeak/1024
    
    if(length(data[[no_algos]]) >= 2){
      for(runs in 2:length(data[[no_algos]])){
        #Algorithm
        algorithm = data[[no_algos]][[runs]]$sub[[1]]$sub[[4]]
        
        #runtime
        tmp_runtimes = cbind(tmp_runtimes,(algorithm$timeEnd - algorithm$timeStart))
        
        #mem
        tmp_mems = cbind(tmp_mems,algorithm$memPeak/1024)
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
  
  n <- length(algorithm_names)
  cols = getDistinctColors(n)
  
  if(sort_by == "names"){
    order_by = order(algorithm_names)
    algorithm_names = algorithm_names[order_by]
    runtimes = runtimes[order_by]
    mems = mems[order_by]
    cols = cols[order_by]
  }else if(sort_by == "runtimes"){
    order_by = order(runtimes)
    algorithm_names = algorithm_names[order_by]
    runtimes = runtimes[order_by]
    mems = mems[order_by]
    cols = cols[order_by]
  }else if(sort_by == "mems"){
    order_by = order(mems)
    algorithm_names = algorithm_names[order_by]
    runtimes = runtimes[order_by]
    mems = mems[order_by]
    cols = cols[order_by]
  }else{
    order_by = 1:length(algorithm_names)
    algorithm_names = algorithm_names[order_by]
    runtimes = runtimes[order_by]
    mems = mems[order_by]
    cols = cols[order_by]
  }
  
  prepare_plot_data(algorithm_names, runtimes, mems, pareto, logarithmic, plot_scatter, memory_additional,cols)
}



prepare_plot_data <- function(names, runtimes, mems, pareto, logarithmic, plot_scatter, memory_additional = T, cols){
  
  label_runtime = "Runtime in milliseconds"
  if(memory_additional){
    label_mem = "Additional memory peak"
  }else{
    label_mem = "Total memory peak"
  }
  
  label_main = "Memory & runtime measurements"
  header_name = paste(" (",args[3],sep = "")
  
  
  size = as.numeric(args[4])
  label_unit = "Bytes"
  if(size >= 1024 * 1024 * 1024){
    size = round(size / 1024 / 1024 / 1024, digits = 2)
    label_unit = "GiB"
  }else if(size >= 1024 * 1024){
    size = round(size / 1024 / 1024, digits = 2)
    label_unit = "MiB"
  }else if(size >= 1024){
    size = round(size / 1024, digits = 2)
    label_unit = "KiB"
  } 
  
  header_name = paste(header_name, ", size: ", size, label_unit,")",sep = "");
  
  
  #If values are too big -> next unit
  if(max(runtimes)>1000*60){
    runtimes = runtimes/(1000*60)
    label_runtime = "Runtime in minutes"
  }else if(max(runtimes) > 100){
    runtimes = runtimes/1000
    label_runtime = "Runtime in seconds"
  }
  if(max(mems) > 10*1024*1024){
    mems = mems / 1024 / 1024
    label_mem = paste(label_mem, "in GiB")
  }else if(max(mems) > 10*1024){
    mems = mems / 1024
    label_mem = paste(label_mem, "in MiB")
  }else{
    label_mem = paste(label_mem, "in KiB")
  }
  
  for(i in 1:length(names)){
    if(nchar(names[i]) > 12){
      names[i] = paste(substring(names[i],1,8),"...", substring(names[i],nchar(names[i])-2,nchar(names[i])),sep = "")
    }
  }
  
  if(plot_scatter == F){
    plot_benchmark_multi_bar(names, runtimes, mems, label_runtime,
                           label_mem, paste(label_main,header_name, sep=""), cols)
  }else{
  
    if(logarithmic){
      #remove data of Naiv
      without_naive = which(names=="Naiv")
      names = names[-without_naive]
      runtimes = runtimes[-without_naive]
      mems = mems[-without_naive]
      cols = cols[-without_naive]
      
      label_mem = paste(label_mem, "(logarithmic scale)", sep = " ")
      label_runtime = paste(label_runtime, "(logarithmic scale)", sep = " ")
      logarithmic = "xy"
    }else{
      logarithmic = ""
    }
    
    if(pareto){
      #remove data of Naiv
      without_naive = which(names=="Naiv")
      names = names[-without_naive]
      runtimes = runtimes[-without_naive]
      mems = mems[-without_naive]
      cols = cols[-without_naive]
      
      algo_data = cbind(mems, runtimes)
      is_pareto = calculate_paretofront(algo_data)
      pareto_inidices = which(is_pareto)
      
      names = names[pareto_inidices]
      runtimes = runtimes[pareto_inidices]
      mems = mems[pareto_inidices]
      cols = cols[pareto_inidices]
      
      if(max(mems) < 1){
        mems = mems * 1024
        label_main =  gsub("MiB ", "KiB", label_main)
        label_main =  gsub("GiB ", "MiB", label_main)
      }

      label_main = paste(label_main, "- Paretofront", sep = " ")
    }
    
    
    plot_benchmark_multi_scatter(names, runtimes, mems, logarithmic,
                                 label_runtime, label_mem,
                                 paste(label_main, header_name,sep = ""), cols)
  }
}

#testList= list(name="a",x=1,y=2)
#testList2= list(name="b",x=2,y=1)

#datafra=data.frame(name=c("d","e"),x=c(3,2),y=c(1,5),
#                   stringsAsFactors = FALSE)

pdf(paste(args[1],".pdf")) 


if(args[2] == 0){
  extract_details(args[1])
}else{
  #Bar plots with total memory peaks
  extract_all_stats(args[1], plot_scatter = F, sort_by = "names", memory_additional = F)
  extract_all_stats(args[1], plot_scatter = F, sort_by = "runtimes", memory_additional = F)
  extract_all_stats(args[1], plot_scatter = F, sort_by = "mems", memory_additional = F)
  
  #Scatter plots with total memory peaks
  extract_all_stats(args[1], plot_scatter = T, memory_additional = F)
  extract_all_stats(args[1], pareto = T, plot_scatter = T, memory_additional = F)
  extract_all_stats(args[1], logarithmic = T, plot_scatter = T, memory_additional = F)
}

maxfreq=paste(as.numeric(try(substring(system("lscpu | grep 'CPU max MHz'",
                                              intern=TRUE),24,27)))/1000,"GHz")
name = try(substring(system("lscpu | grep 'Model name:'",intern=TRUE),24))
ram = try(substring(system("grep MemTotal /proc/meminfo",intern=TRUE),18))
ram = paste(round(as.numeric(try(substring(ram,0,nchar(ram)-3)))/1000/1000, digits = 0), "GB");

name = paste(name, ", ", maxfreq, " max", sep = "")
ram = paste(ram, "RAM")
text_input = paste("Input file:",args[3])
size = as.numeric(args[4])

label_unit = "Bytes"
if(size >= 1024 * 1024 * 1024){
  size = round(size / 1024 / 1024 / 1024, digits = 2)
  label_unit = "GiB"
}else if(size >= 1024 * 1024){
  size = round(size / 1024 / 1024, digits = 2)
  label_unit = "MiB"
}else if(size >= 1024){
  size = round(size / 1024, digits = 2)
  label_unit = "KiB"
} 
text_size = paste("Size: ", size, label_unit, sep = "")

plot.new()
legend("top", legend = list(name, ram, text_input, text_size),
       col = 1, pch = 15, xpd = TRUE, bty = "n")
title("Experimental setup", line=1)



dev.off()

#for command line:
#R -e 'install.packages("package", repos="http://cran.us.r-project.org")'
