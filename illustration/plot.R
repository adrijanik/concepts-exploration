

library("tidyverse")
library("reshape2")
theme_set(theme_bw() + theme(panel.grid = element_blank()))

#' read in data
iris <- read_csv("iris.csv") %>%
  rename(class = `0`, sepal_length = `1`, sepal_width = `2`, petal_length = `3`, petal_width = `4`)
iris$class <- as.factor(iris$class)

w1 <- read_csv("w1.csv")
colnames(w1) <- paste0("w_k", 1:ncol(w1))
b1 <- read_csv("b1.csv") %>%.[[1]]

sigm <- function(x) {
  exp(-x) / (1 + exp(-x))
}

#' let's visualize this first logistic regression
#' first just comparing the first and second dimensions of iris
plot(iris$petal_length, iris$petal_width)

K <- nrow(w1)
u_i <- seq(-3, 3, length.out = 100)
u <- as.matrix(expand.grid(u_i, u_i))
u <- cbind(u, 0, 0)

for (k in seq_len(K)) {
  probs[[k]] <- data.frame(
    w_k = k,
    u = u,
    p = sigm(u %*% t(as.matrix(w1[k, ])))
  )
  colnames(probs[[k]]) <- c("w_k", colnames(iris[, -1]), "p")
}

probs <- do.call(rbind, probs)

ggplot(probs %>% filter(w_k < 10)) +
  geom_point(
    data = iris,
    aes(x = sepal_length, y = sepal_width, col = class)
  ) +
  geom_tile(
    aes(x = sepal_length, y = sepal_width, fill = p),
    alpha = 0.5
  ) +
  scale_fill_gradient2(midpoint = 0.5) +
  facet_wrap(~ w_k)


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
