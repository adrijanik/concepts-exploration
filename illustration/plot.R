#' Plotting Concepts in a Toy Example
#'
#' The main plot here is a plot of the different hyperplanes learned in the
#' first layer, along with the concept scores obtained when you perturb each
#' point's representation to move slightly in the direction orthogonal to that
#' hyperplane.
library("ggplot2")
library("dplyr")
library("reshape2")
theme_set(theme_bw() + theme(panel.grid = element_blank()))

#' read in data
iris <- read_csv("iris.csv") %>%
  rename(class = `0`, sepal_length = `1`, sepal_width = `2`, petal_length = `3`, petal_width = `4`)
iris$sample <- 0:(nrow(iris) - 1)
iris$class <- as.factor(iris$class)

sigm <- function(x) {
  exp(-x) / (1 + exp(-x))
}

#' Evaluate the logistic regression probabilities across the hidden layer units
K <- nrow(w1)
u_i <- seq(-3, 3, length.out = 100)
u <- as.matrix(expand.grid(u_i, u_i))
u <- cbind(0, u, 0)

probs <- list()
for (k in seq_len(K)) {
  probs[[k]] <- data.frame(
    w_k = k - 1,
    u = u,
    p = sigm(u %*% t(as.matrix(w1[k, ])))
  )
  colnames(probs[[k]]) <- c("w_k", c("sepal_length", "sepal_width", "petal_length", "petal_width"), "p")
}

probs <- do.call(rbind, probs)

#' Plot the concept scores across first 5 h's
scores <- read_csv("scores.csv")
colnames(scores) <- c(paste0("class_", 0:2), "sample", "w_k")
scores <- melt(scores, id.vars = c("sample", "w_k"), value.name = "score", variable.name = "logit")

concept_scores <- iris %>%
  left_join(scores) %>%
  filter(w_k < 5)

ggplot(concept_scores) +
  geom_tile(
    data = probs %>% filter(w_k < 5),
    aes(x = sepal_width, y = petal_length, fill = p),
    alpha = 0.5
  ) +
  geom_point(
    aes(x = sepal_width, y = petal_length, col = score, shape = class),
    size = 0.8
  ) +
  scale_fill_gradient2(midpoint = 0.5, low = "white", high = "black") +
  scale_color_gradient2(midpoint = 0, low = "indianred", high = "royalblue", mid = "black") +
    facet_grid(w_k ~ logit)

#' The decision boundary, when we set the two unplotted variables to 0
p_hat <- read_csv("eval_pts.csv")
p_hat[, 1:3] <- exp(p_hat[, 1:3]) / rowSums(exp(p_hat[, 1:3]))
p_hat <- p_hat %>%
  melt(id.vars = c("x0", "x1"))

ggplot(p_hat) +
  geom_tile(
    aes(x = x0, y = x1, fill = value)
  ) +
  geom_point(
    data = iris,
    aes(x = sepal_width, y = petal_length, col = class)
  ) +
  scale_fill_gradient2(midpoint = 0.5, low = "white", high = "black") +
  facet_wrap(~ variable)
