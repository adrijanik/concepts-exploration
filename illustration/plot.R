#' Plotting Concepts in a Toy Example
#'
#' The main plot here is a plot of the different hyperplanes learned in the
#' first layer, along with the concept scores obtained when you perturb each
#' point's representation to move slightly in the direction orthogonal to that
#' hyperplane.
library("ggplot2")
library("dplyr")
library("reshape2")
library("readr")
theme_set(theme_bw() + theme(panel.grid = element_blank()))
sigm <- function(x) {
  exp(-x) / (1 + exp(-x))
}

#' read in data
combined <- read_csv("combined.csv") %>%
  mutate(y = as.factor(y))
p_cols <- paste0("p", 0:2)
combined[ p_cols] <- exp(combined[, p_cols]) / rowSums(exp(combined[, p_cols]))

w1 <- as.matrix(read_csv("w1.csv"))
b1 <- as.matrix(read_csv("b1.csv"))

#' Evaluate the logistic regression probabilities across the hidden layer units
K <- nrow(w1)
u_i <- seq(-1, 1, length.out = 100)
u <- as.matrix(expand.grid(u_i, u_i))

probs <- list()
for (k in seq_len(K)) {
  probs[[k]] <- data.frame(
    w_k = k,
    u = u,
    p = sigm(u %*% w1[k, ] + b1[k, ])
  )
  colnames(probs[[k]]) <- c("w_k", "X1", "X2", "p")
}
probs <- do.call(rbind, probs)

#' Building unified concepts data
w_order <- hclust(dist(w1))$order

scores <- read_csv("scores.csv") %>%
  mutate(index = as.factor(index + 1))
colnames(scores)[1:3] <- paste0("class_", 0:2)
scores <- scores %>%
  melt(id.vars = c("sample", "index"), value.name = "score", variable.name = "class") %>%
  mutate(index = factor(index, levels = w_order))
concept_scores <- combined %>%
  mutate(sample = 0:(n() - 1),) %>%
  select(sample, y, starts_with("X"), starts_with("p"), starts_with("h")) %>%
  left_join(scores)

cur_levels <- c("7", "18", "12", "2")

w1 <- data.frame(w1) %>%
  mutate(index = factor(1:n(), levels = w_order)) %>%
  rename(wj1 = X0, wj2 = X1)

concept_scores <- concept_scores %>%
  left_join(w1)

mw1 <- w1 %>%
  melt(id.vars = "index", variable.name = "j", value.name = "wj")

#' The decision boundary, when we set the two unplotted variables to 0
p_hat <- read_csv("eval_pts.csv")
p_hat[, 1:3] <- exp(p_hat[, 1:3]) / rowSums(exp(p_hat[, 1:3]))
p_hat <- p_hat %>%
  melt(id.vars = c("x0", "x1"))

#' Scores in random directions
scores_rand <- read_csv("scores_rand.csv") %>%
  melt(id.vars = c("sample", "index"), value.name = "score", variable.name = "class")

scores_rand_v <- scores_rand %>%
  group_by(index, class) %>%
  summarise(sigma2 = var(score)) %>%
  arrange(desc(sigma2))

#' scores from random positions
scores_cluster <- read_csv("scores_cluster.csv") %>%
  melt(id.vars = c("sample", "k"), value.name = "score", variable.name = "class")

scores_cluster_v <- scores_cluster %>%
  group_by(k, class) %>%
  summarise(sigma2 = var(score))

#' fdr result on these scores
locfdr(scores_rand_v$sigma2, nulltype = 1)
locfdr(scores_cluster_v$sigma2, nulltype = 1)

#' all the plots
ggplot(concept_scores %>%
       filter(index %in% cur_levels)) +
  geom_tile(
    data = probs %>% rename(index = w_k) %>% filter(index %in% cur_levels),
    aes(x = X1, y = X2, fill = p)
  ) +
  geom_point(
    aes(
      x = X1, y = X2, col = score
    ),
    size = 0.3
  ) +
  scale_color_gradient2() +
  scale_fill_gradient2(low = "white", high = "black") +
  facet_grid(class ~ index)

ggplot(concept_scores) +
  geom_histogram(
    aes(
      x = score,
      fill = class
    ),
    position = "identity",
    bins = 50,
    alpha = 0.2
  ) +
  ylim(0, 40) +
  facet_wrap(~ index)

ggplot(w1) +
  geom_text(
    aes(x = wj1, y = wj2, label = index)
  ) +
  coord_fixed()

ggplot(p_hat) +
  geom_tile(
    aes(x = x0, y = x1, fill = value)
  ) +
  geom_point(
    data = combined,
    aes(x = X1, y = X2, col = y),
    size = 0.6
  ) +
  scale_color_brewer(palette = "Set2") +
  scale_fill_gradient2(midpoint = 0.5, low = "white", high = "black") +
  coord_fixed() +
  facet_wrap(~ variable)

ggplot(scores_rand %>% sample_n(10000)) +
  geom_point(
    aes(x = reorder(index, score, var), y = score, col = as.factor(class)),
    alpha = 0.2
  ) +
  scale_color_brewer(palette = "Set2") +
  theme(axis.text.x = element_blank())

ggplot(scores_cluster) +
  geom_point(
    aes(x = reorder(k, score, var), y = score, col = as.factor(class)),
    alpha = 0.6
  ) +
  scale_color_brewer(palette = "Set2") +
  theme(axis.text.x = element_blank())

ggplot(concept_scores %>% filter(class == "class_0")) +
  geom_point(
    aes(x = h5, y = p0, col = score)
  ) +
  scale_color_gradient2()

ggplot(concept_scores %>% filter(class == "class_0")) +
  geom_point(
    aes(x = h5, y = h0, col = score, shape = y, size = p0)
  ) +
  scale_size(range = c(0.02, 2)) +
  scale_color_gradient2(mid = "#9F9F9F")

ggplot(concept_scores %>% filter(class == "class_0")) +
  geom_point(
    aes(x = X1, y = h5, col = score, size = p0)
  ) +
  scale_size(range = c(0.02, 2)) +
  scale_color_gradient2(mid = "#9F9F9F")

ggplot(mw1) +
  geom_tile(
    aes(x = j, y = index, fill = wj)
  ) +
  scale_fill_gradient2(high = "royalblue", low = "indianred")
