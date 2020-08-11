library(readr)
CD <- read_csv(file.choose())
View(CD)
class(CD)
dim(CD)
str(CD)

summary(CD)

plot(CD)

attach(CD)

hist(X1)
hist(price) # Price is postively Skrewed with a long tail towords the right
hist(speed)
hist(hd)
plot(price,speed)
qqplot(price, speed)
plot(price ~ speed+hd+ram+screen)

chisq.test(table(price, speed)) 
qqplot(price,speed)
qqline(price, probs = c(0.25,0.75))

