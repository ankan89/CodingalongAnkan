

library(readxl)

mob <- read_excel(file.choose())

View(mob)

summary(mob)

str(mob)

mob1 <- data.frame(mob)
colnames(mob1) <- c("MobileNo")
?colnames
?substring
library(numbers)

a <- div(mob1$MobileNo,10^9)
a1 <- mod(mob1$MobileNo, 10^9)

b <- div(a1,10^8)
b1 <- mod(a1, 10^8) 

c <- div(b1,10^7)
c1 <- mod(b1, 10^7)

d <- div(c1,10^6)
d1 <- mod(c1, 10^6)

e <- div(d1,10^5)
e1 <- mod(d1, 10^5)

f <- div(e1,10^4)
f1 <- mod(e1, 10^4)

g <- div(f1,10^3)
g1 <- mod(f1, 10^3)

h <- div(g1,10^2)
h1 <- mod(g1, 10^2)

i <- div(h1,10^1)
i1 <- mod(h1, 10^1)


mob2 <- cbind(a,b,c,d,e,f,g,h,i,i1)

mob1 <- cbind.data.frame(mob1,mob2)



colnames(mob1) <- c("Customer Mob Nos","1st","2nd","3rd","4th","5th","6th","7th","8th","9th","10")

str(mob1)

attach(mob1)

j <- 10  # Trump Variable

k <- (a+b+c+d+e+f+g+h+i)*j

New_Tarrif <- k


mob1 <- cbind.data.frame(mob1,New_Tarrif)

View(mob1)

write.csv(mob1, file = "mob1.csv")
