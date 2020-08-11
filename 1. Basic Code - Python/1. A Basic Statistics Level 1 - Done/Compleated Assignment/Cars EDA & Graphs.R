library(readr)

cars <- read_csv("/Users/Ankan/PycharmProjects/Basiccode/1. Assignments/1. A Basic Statistics Level 1 - Done/Cars.csv")

View(cars)

attach(cars)

table(cars)

summary(cars)

str(cars)

par(mfrow = c(3,2))
pairs(cars)
qqplot(MPG,HP)
qqline(MPG)
qqplot(MPG,VOL)
qqline(MPG)
qqplot(MPG,SP)
qqline(MPG)
qqplot(MPG,WT)
qqline(MPG)
library(lattice)

par(mfrow = c(3,2))
hist(HP)
hist(MPG)
hist(SP)
hist(VOL)
hist(WT)
stars (cars, draw.segments = TRUE, Key.loc = c(13,1.5))


par(mfrow = c(3,2))
boxplot(HP, horizontal = T)
boxplot(MPG, horizontal = T)
boxplot(SP, horizontal = T)
boxplot(VOL, horizontal = T)
boxplot(WT, horizontal = T)

stem (HP)
