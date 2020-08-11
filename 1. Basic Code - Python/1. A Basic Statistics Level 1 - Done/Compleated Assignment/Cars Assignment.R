
library(readr)
View(mtcars)

cars <-  read_csv(file.choose())

newdata <- cars

attach(newdata)

class(newdata)
View(newdata)

length(newdata)

nrow(newdata)
ncol(newdata)
dim(newdata)

str(newdata)

summary(newdata)



shapiro.test(MPG) 
shapiro.test(log10(HP))
shapiro.test(VOL)
shapiro.test(SP)
shapiro.test(WT)

a <- stack(cars)
View(a)

var.test( MPG,HP)
t.test( MPG, HP, alternative = "two.sided", conf.level = 0.95, correct = TRUE)
t.test( MPG, HP, alternative = "greater", var.equal = T) 


var.test( MPG,VOL)
t.test( MPG,VOL, alternative = "two.sided", conf.level = 0.95, correct = TRUE)
t.test( MPG,VOL, alternative = "greater", var.equal = T) 


var.test( MPG,SP)
t.test( MPG, SP, alternative = "two.sided", conf.level = 0.95, correct = TRUE)
t.test( MPG, SP, alternative = "greater", var.equal = T) 


sd <- sd(MPG)

qqnorm(newdata$MPG)
qqline(newdata$MPG)




newdata1 <- log(MPG)
attach(newdata1)


par(mfrow = c(3,2))
hist(MPG, col = "red", xlab = "MPG") # It gives a sence that the output variable might be normally distributed.
hist(HP, col = "deepskyblue1", xlab = "HP" ) #It is positively skewed and might not ontains any outliers.
hist(VOL, col = "darkorchid3", xlab = "VOL" ) #It might follow the normal distribution.
hist(SP, col = "mediumspringgreen", xlab = "SP" ) # The variables might be normally distributed as it some positive shewness which coould be because of some outlier
hist(WT, col = "mintcream", xlab = "WT" ) # The variables might not be normally distributed as there are some hints of outliers.

library(corrgram)
corrgram(newdata1) 
?corrgram

boxplot(MPG, col = "red", xlab = "MPG", horizontal = T) # It  again gives a indication that the output variable might be normally distributed.
boxplot(HP, col = "deepskyblue1", xlab = "HP", horizontal = T ) #It is positively skewed and with outliers.
boxplot(VOL, col = "darkorchid3", xlab = "VOL", horizontal = T ) #It might follow the normal distribution.
boxplot(SP, col = "mediumspringgreen", xlab = "SP", horizontal = T ) # The variables might be normally distributed as it has positive shewness with a bit of outlier
boxplot(WT, col = "mintcream", xlab = "WT", horizontal = T ) # The variables normal distribution could be affected by a outliers.

?pairs
library(graphics)
pairs(newdata1) # HP~SP & VOL~WT are strongly corelated, so we need to work on come transfermation. 


?scatter

plot(MPG,HP)
plot(MPG,VOL)
plot(MPG,SP)
plot(MPG,WT)

cor(newdata)

library(corpcor)

library(moments)

Q9 <- read_csv(file.choose())

skewness(Q9$speed)
skewness(Q9$dist)

kurtosis(Q9$speed)
kurtosis(Q9$dist)

Q9b <- read_csv(file.choose()) 

skewness(Q9b$SP)
skewness(Q9b$WT)

kurtosis(Q9b$SP)
kurtosis(Q9b$WT)




n <- 2000
s <-  30 
xbar <-  200

error <-  qnorm(0.99)*s/sqrt(n)

xbar - error
xbar + error


mean <- mean(MPG)
sd <- sd(MPG)

pnorm(38,mean,sd)
# We got prob. less than 38% so to get prob. >38%
1-pnorm(38,mean,sd) #20(a) Ans
pnorm(40,mean,sd) #20(b) Ans
pnorm(50,mean,sd) - pnorm(20,mean,sd) #20(c) Ans






qqnorm(MPG)
qqline(MPG)




WSAT <-  read_csv(file.choose())
attach(WSAT)
qqnorm(Waist)
qqline(Waist)
qqnorm(AT)
qqline(AT)




















