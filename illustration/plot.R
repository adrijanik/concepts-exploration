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
library("locfdr")
theme_set(theme_bw() + theme(panel.grid = element_blank()))
sigm <- function(x) {
  exp(-x) / (1 + exp(-x))
}

negvar <- function(x) {
  -var(x)
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
    p = (u %*% w1[k, ] + b1[k, ]) * as.numeric(I((u %*% w1[k, ] + b1[k, ]) > 0))
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

cur_levels <- c("7", "8", "1")

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
  summarise(sigma2 = sd(score)) %>%
  arrange(desc(sigma2))

#' scores from random positions
scores_cluster <- read_csv("scores_cluster.csv") %>%
  melt(id.vars = c("sample", "k"), value.name = "score", variable.name = "class")

scores_cluster_v <- scores_cluster %>%
  group_by(k, class) %>%
  summarise(sigma2 = var(score))

#' all the plots
ggplot(concept_scores %>%
       filter(index %in% cur_levels)) +
  geom_tile(
    data = probs %>% rename(index = w_k) %>% filter(index %in% cur_levels),
    aes(x = X1, y = X2, fill = p)
  ) +
  geom_point(
    aes(
      x = X1, y = X2, col = y, alpha = abs(score), size = abs(score)
    ),
  ) +
  labs(
    "size" = "Absolute Concept Activation",
    "alpha" = "Absolute Concept Activation",
    "color" = "Class",
    "fill" = "Learned Feature"
  ) +
  scale_y_continuous(expand = c(0, 0)) +
  scale_x_continuous(expand = c(0, 0)) +
  scale_color_brewer(palette = "Set2") +
  scale_alpha(range = c(0.2, 1)) +
  scale_size(range = c(0.1, 2)) +
  scale_fill_gradient2(low = "white", high = "black") +
  facet_grid(class ~ index) +
  coord_fixed() +
  theme(legend.position = "bottom", legend.direction = "vertical")

ggsave("activations_with_surface.png", width = 5, height = 8, dpi = 300)

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
    size = 0.3, alpha = 0.6
  ) +
  labs(
    "color" = "Class",
    "fill" = "Predicted Probability"
  ) +
  scale_color_brewer(palette = "Set2") +
  guides(color = guide_legend(override.aes = list(size = 2, alpha = 1))) +
  scale_fill_gradient2(midpoint = 0.5, low = "white", high = "black") +
  coord_fixed() +
  facet_wrap(~ variable) +
  theme(legend.position = "bottom")

ggsave("p_hat_surface.png", width = 5, height = 3, dpi=250)

ggplot(scores_rand %>% sample_n(10000)) +
  geom_point(
    aes(x = reorder(index, score, var), y = score, col = as.factor(class)),
    alpha = 0.2
  ) +
  scale_color_brewer(palette = "Set2") +
  theme(axis.text.x = element_blank())

ggplot(scores_cluster) +
  geom_point(
    aes(x = reorder(k, score, negvar), y = score, col = as.factor(class)),
    alpha = 0.6, size = 0.9,
    position = position_jitter(0.2, 0.001)
  ) +
  labs(
    "x" = "Cluster",
    "y" = "Concept Activation",
    "col" = "Class"
  ) +
  facet_wrap(~ class) +
  scale_color_brewer(palette = "Set2") +
  theme(axis.text.x = element_blank(), legend.position = "bottom", axis.ticks = element_blank())

ggsave("cluster_by_activation.png", dpi = 400)

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

ggplot(scores_rand_v) +
  geom_histogram(aes(x = sigma2), bins = 100)

fdr_fit <- locfdr(scores_rand_v$sigma2)
scores_rand_v$fdr <- fdr_fit$fdr

joined <- scores_rand %>%
  left_join(scores_rand_v)

ggplot(joined %>% sample_n(10000)) +
  geom_point(
    aes(x = reorder(index, score, negvar), y = score, col = as.factor(class), alpha = fdr, size = fdr),
    position = position_jitter(0, 0.04)
  ) +
  scale_alpha(range = c(1, 0.1)) +
  scale_size(range = c(4, 0.2)) +
  scale_color_brewer(palette = "Set2") +
  theme(axis.text.x = element_blank()) +
  labs(
    x = "Random Direction",
    y = "Concept Activation Score",
    size = "False Discovery Rate",
    alpha = "False Discovery Rate",
    color = "Class"
  ) +
  theme(legend.position =  "bottom", axis.ticks = element_blank())


ggsave("fdr_by_activation.png", dpi = 400)


vs <- read_csv("v_rand.csv")

directions <- list()
for (i in seq_len(nrow(vs))) {
  directions[[i]] <- t(as.matrix(w1[, 1:2])) %*% as.numeric(vs[i, ])
  directions[[i]] <- directions[[i]] / sqrt(sum(directions[[i]] ** 2))
}
directions <- t(do.call(cbind, directions))
directions <- data.frame(directions)

for (k in 0:2) {
  directions[[paste0("fdr_", k)]] <- scores_rand_v %>%
    filter(class == k) %>%
    arrange(index) %>%
    .[["fdr"]]
}

directions <- directions %>%
  melt(
    measure.vars = c("fdr_0", "fdr_1", "fdr_2"),
    value.name =  "fdr",
    variable.name = "class"
  )

ggplot(directions) +
  geom_segment(
    aes(x = 0, y = 0, xend = wj1, yend = wj2, alpha = fdr),
    size = 0.5
  ) +
  geom_point(
    data = combined,
    aes(x = X1, y = X2, col = y),
    size = 0.2
  ) +
  geom_point(
    aes(x = wj1, y = wj2, alpha = fdr, size = fdr)
  ) +
  labs(
    col = "Class",
    alpha = "Local FDR",
    size = "Local FDR"
  ) +
  guides(color = guide_legend(override.aes = list(size = 2, alpha = 1))) +
  facet_grid(. ~ class) +
  scale_color_brewer(palette = "Set2") +
  scale_alpha(range =  c(1, 0.1)) +
  scale_size(range =  c(4, 0.2)) +
  coord_fixed() +
  theme(legend.position = "bottom")

ggsave("fdr_directions.png", width = 6, height = 4)


centroids <- read_csv("cluster_centroids.csv")
colnames(centroids) <- c("c1", "c2")
centroids$k <- 0:(nrow(centroids) - 1)

scores_cluster <- scores_cluster %>%
  left_join(centroids)

scores_cluster_means <- scores_cluster %>%
  group_by(k, class, c1, c2) %>%
  summarise(mu_s = mean(score), sigma_s = sd(score))

combined$sample <- 0:(nrow(combined) - 1)

ggplot(scores_cluster_means) +
  geom_tile(
    data = probs %>% filter(w_k == 1),
    aes(x = X1, y = X2, fill = p)
  ) +
  geom_point(
    aes(x = c1, y = c2, size = sigma_s),
    shape = 15, col = "#9955bb"
  ) +
  geom_point(
    data = combined %>% left_join(scores_cluster),
    aes(x = X1, y = X2, col = y, alpha = abs(score)),
    size = 1.2
  ) +
  labs(
    x = "x",
    y = "y",
    col = "Class",
    size = "SD within Cluster",
    fill = "Learned Feature",
    alpha = "Absolute Concept Activation"
  ) +
  scale_alpha(range = c(0.05, 1), breaks = c(0.1, 0.25, 0.5)) +
  scale_x_continuous(expand = c(0, 0)) +
  scale_y_continuous(expand = c(0, 0)) +
  facet_grid(. ~ class) +
  scale_size(range = c(.3, 4)) +
  coord_fixed() +
  scale_color_brewer(palette = "Set2") +
  scale_fill_gradient2(low = "white", high = "black") +
  guides(fill = guide_colorbar(barheight = 2.5)) +
  theme(
    legend.position = "bottom", legend.direction = "vertical"
  )

ggsave("cluster_activation_sds.png", width = 7, height = 4)
