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

plot_benchmark_single<-function(names, runtimes, mems)
{
  par(mfrow=c(1,2),mai=c(0.5,1,1,1))
  barplot(as.matrix(runtimes),beside=FALSE,col = 2:(length(runtimes)+1))
  title("Runtime", line=1)
  barplot(mems,col = 2:(length(mems)+1))
  title("Memory Usage", line=1)
  mtext(name, side=3, outer=TRUE, line=-2,cex = 2)
}
extract_details<-function(json_name)
{
  data= fromJSON(file=json_name,simplify = TRUE)
  #Algorithm
  algorithm= data$sub[1][[1]]$sub[[4]]
  #Phasen
  phasen= algorithm$sub
}



#testList= list(name="a",x=1,y=2)
#testList2= list(name="b",x=2,y=1)

#datafra=data.frame(name=c("d","e"),x=c(3,2),y=c(1,5),stringsAsFactors = FALSE)
