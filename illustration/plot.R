

library("tidyverse")
library("reshape2")

#' read in data
iris <- read_csv("iris.csv") %>%
  rename(class = `0`, sepal_length = `1`, sepal_width = `2`, petal_length = `3`, petal_width = `4`)
iris$class <- as.factor(iris$class)

w1 <- read_csv("w1.csv")
colnames(w1) <- paste0("w_k", 1:ncol(w1))
b1 <- read_csv("b1.csv") %>%.[[1]]

#' let's visualize this first logistic regression
#' first just comparing the first and second dimensions of iris
plot(iris$sepal_length, iris$sepal_width)
for (k in seq_len(20)) {
  plane_1 <- data.frame(
    u = seq(-5, 5, length.out = 10)
  )
  plane_1$fu = - (b1[k] + w1[[k, 1]] * plane_1$u) / w1[[k, 2]]
  lines(plane_1$u, plane_1$fu)
}

#' now the parallel coordinates view
miris <- iris %>%
  mutate(sample = row_number()) %>%
  melt(id.vars = c("sample", "class"))

ggplot(miris) +
  geom_line(aes(x = variable, y = value, col = class, group = sample))

#' Parallel coordinates
ggplot(miris) +
  geom_line(aes(x = variable, y = value, col = class, group = sample))

#' random vectors v on the hyperplane wk^T v = -bk
p <- ncol(iris) - 1
k <- 1
w1k <- unlist(w1[k, ])
H_perp <- diag(p) - w1k %*% t(w1k) / sum(w1k ^ 2)

B <- 100
v <- matrix(rnorm(B * p), B, p) %*% H_perp
colnames(v) <- colnames(iris[, -1])
for (b in seq_len(B)) {
  v[b, ] <- v[b, ] - b1[k] * w1k / sum(w1k ^ 2)
}

mv <- melt(v)
ggplot(miris) +
  geom_line(aes(x = variable, y = value, col = class, group = sample)) +
  geom_line(data = mv, aes(x = Var2, y = value, group = Var1), alpha = 0.1) +
  facet_grid(class ~ .)
