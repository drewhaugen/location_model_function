build_location_tree <- function(pitch_type_name, pitch_data){
  # Load Libraries
  library(magrittr)
  library(tidymodels)
  
  pitch_data <- pitch_data%>%
    dplyr::filter(!is.na(plate_x),
                  !is.na(plate_z),
                  !is.na(balls),
                  !is.na(strikes),
                  !is.na(stand),
                  !is.na(p_throws),
                  !is.na(description),
                  !balls == 4,
                  !strikes == 3)%>%
    dplyr::mutate(balls = as.factor(balls),
                  strikes = as.factor(strikes),
                  stand = as.factor(stand),
                  p_throws = as.factor(p_throws),
                  swing = ifelse(description %in% c("foul_tip", "swinging_strike",  
                                                    "swinging_strike_blocked", 
                                                    "missed_bunt",
                                                    "foul", "hit_into_play", 
                                                    "foul_bunt", "bunt_foul_tip"),
                                 "swing", "take"))%>%
    dplyr::mutate(swing = as.factor(swing))
  
  # Split into Training and Testing
  set.seed(456)
  swing_split <- initial_split(pitch_data, strata = swing)
  swing_train <- training(swing_split)
  swing_test  <- testing(swing_split)
  
  # Swing Model Spec
  swing_spec <- boost_tree(
    trees = 100, 
    tree_depth = tune(), min_n = tune(), 
    sample_size = tune(), mtry = tune(),         
    learn_rate = 0.1) %>% 
    set_engine("xgboost") %>% 
    set_mode("classification")
  
  # Swing Recipe
  swing_rec <- recipe(swing ~ plate_x + plate_z + balls + strikes + 
                        + p_throws + stand, data = swing_train)%>%
    step_dummy(balls, strikes, stand, p_throws, one_hot = T)
  
  # Swing Grid
  swing_grid <- grid_latin_hypercube(
    tree_depth(),
    min_n(),
    sample_size = sample_prop(),
    finalize(mtry(), pitch_data),
    size = 10)
  
  # Swing Workflow
  swing_wf <- workflow()%>%
    add_recipe(swing_rec)%>%
    add_model(swing_spec)
  
  # Swing Folds
  set.seed(123)
  swing_folds <- vfold_cv(swing_train, v = 5, strata = swing)
  
  # Tune Swing Model
  set.seed(234)
  doParallel::registerDoParallel()
  swing_tune <- tune_grid(
    swing_wf,
    resamples = swing_folds,
    grid = swing_grid,
    control = control_grid(save_pred = TRUE))
  
  # Finalize Swing Workflow
  final_swing <- finalize_workflow(
    swing_wf,
    select_best(swing_tune, "roc_auc"))
  
  # Fit Workflow
  swing_fit <- 
    final_swing %>%
    last_fit(swing_split) 
  
  swing_model <- butcher::axe_env.workflow(extract_workflow(swing_fit))
  
  # Save Swing Probability Model
  swing_mod_name <- paste0(pitch_type_name, "_", "loc_swing_mod.rds")
  readr::write_rds(swing_model, swing_mod_name)
  
  # Create Take Dataframe
  take_data <- pitch_data%>%
    dplyr::filter(swing == "take")%>%
    dplyr::mutate(outcome = ifelse(description == "called_strike", "called_strike",
                                   ifelse(description %in% c("ball", 
                                                             "blocked_ball", 
                                                             "pitchout",
                                                             "foul_pitchout"),
                                          "ball", "hit_by_pitch")))%>%
    dplyr::filter(!is.na(outcome))%>%
    dplyr::mutate(outcome = as.factor(outcome))
  
  # Split into Training and Testing
  set.seed(345)
  take_split <- initial_split(take_data, strata = outcome)
  take_train <- training(take_split)
  take_test  <- testing(take_split)
  
  # Take Model Spec
  take_spec <- boost_tree(
    trees = 100, 
    tree_depth = tune(), min_n = tune(), 
    sample_size = tune(), mtry = tune(),         
    learn_rate = 0.1) %>% 
    set_engine("xgboost", num_class = 3) %>% 
    set_mode("classification")
  
  # Take Recipe
  take_rec <- recipe(outcome ~ plate_x + plate_z + balls + strikes + 
                        + p_throws + stand, data = take_train)%>%
    step_dummy(balls, strikes, stand, p_throws, one_hot = T)
  
  # Take Grid
  take_grid <- grid_latin_hypercube(
    tree_depth(),
    min_n(),
    sample_size = sample_prop(),
    finalize(mtry(), take_train),
    size = 10)
  
  # Take Workflow
  take_wf <- workflow()%>%
    add_recipe(take_rec)%>%
    add_model(take_spec)
  
  # Take Folds
  set.seed(567)
  take_folds <- vfold_cv(take_train, v = 5, strata = outcome)
  
  # Tune Take Model
  set.seed(789)
  take_tune <- tune_grid(
    take_wf,
    resamples = take_folds,
    grid = take_grid,
    control = control_grid(save_pred = TRUE))
  
  # Finalize Take Workflow
  final_take <- finalize_workflow(
    take_wf,
    select_best(take_tune, "roc_auc"))
  
  # Fit Workflow
  take_fit <- 
    final_take %>%
    last_fit(take_split)
  
  take_model <- butcher::axe_env.workflow(extract_workflow(take_fit))
    
  # Save Take Probability Model
  take_mod_name <- paste0(pitch_type_name, "_", "loc_take_mod.rds")
  readr::write_rds(take_model, take_mod_name)
    
  # Create Contact Dataframe
  contact_data <- pitch_data%>%
    dplyr::filter(swing == "swing")%>%
    dplyr::mutate(contact = ifelse(description %in% c("hit_into_play", "foul",
                                                          "foul_pitchout"),
                                       "contact", "whiff"))%>%
    dplyr::filter(!is.na(contact))%>%
    dplyr::mutate(contact = as.factor(contact))
  
  # Split into Training and Testing
  set.seed(891)
  contact_split <- initial_split(contact_data, strata = contact)
  contact_train <- training(contact_split)
  contact_test  <- testing(contact_split)
  
  # Contact Model Spec
  contact_spec <- boost_tree(
    trees = 100, 
    tree_depth = tune(), min_n = tune(), 
    sample_size = tune(), mtry = tune(),         
    learn_rate = 0.1) %>% 
    set_engine("xgboost") %>% 
    set_mode("classification")
  
  # Contact Recipe
  contact_rec <- recipe(contact ~ plate_x + plate_z + balls + strikes + 
                        + p_throws + stand, data = contact_train)%>%
    step_dummy(balls, strikes, stand, p_throws, one_hot = T)
  
  # Contact Grid
  contact_grid <- grid_latin_hypercube(
    tree_depth(),
    min_n(),
    sample_size = sample_prop(),
    finalize(mtry(), contact_train),
    size = 10)
  
  # Contact Workflow
  contact_wf <- workflow()%>%
    add_recipe(contact_rec)%>%
    add_model(contact_spec)
  
  # Contact Folds
  set.seed(912)
  contact_folds <- vfold_cv(contact_train, v = 5, strata = contact)
  
  # Tune Contact Model
  set.seed(826)
  contact_tune <- tune_grid(
    contact_wf,
    resamples = contact_folds,
    grid = contact_grid,
    control = control_grid(save_pred = TRUE))
  
  # Finalize Contact Workflow
  final_contact <- finalize_workflow(
    contact_wf,
    select_best(contact_tune, "roc_auc"))
  
  # Fit Workflow
  contact_fit <- 
    final_contact %>%
    last_fit(contact_split) 
  
  contact_model <- butcher::axe_env.workflow(extract_workflow(contact_fit))
  
  # Save Contact Model
  contact_mod_name <- paste0(pitch_type_name, "_", "loc_contact_mod.rds")
  readr::write_rds(contact_model, contact_mod_name)
  
  # Create Fair Dataframe
  fair_data <- pitch_data%>%
    dplyr::mutate(contact = ifelse(description %in% c("hit_into_play", "foul",
                                                      "foul_pitchout"),
                                   "contact", "whiff"),
                  fair = ifelse(description == "hit_into_play",
                                "fair", "foul"))%>%
    dplyr::filter(!is.na(contact),
                  !is.na(fair))%>%
    dplyr::filter(contact == "contact")%>%
    dplyr::mutate(fair = as.factor(fair))
  
  # Split into Training and Testing
  set.seed(371)
  fair_split <- initial_split(fair_data, strata = fair)
  fair_train <- training(fair_split)
  fair_test  <- testing(fair_split)
  
  # Fair Model Spec
  fair_spec <- boost_tree(
    trees = 100, 
    tree_depth = tune(), min_n = tune(), 
    sample_size = tune(), mtry = tune(),         
    learn_rate = 0.1) %>% 
    set_engine("xgboost") %>% 
    set_mode("classification")
  
  # Fair Recipe
  fair_rec <- recipe(fair ~ plate_x + plate_z + balls + strikes + 
                          + p_throws + stand, data = fair_train)%>%
    step_dummy(balls, strikes, stand, p_throws, one_hot = T)
  
  # Fair Grid
  fair_grid <- grid_latin_hypercube(
    tree_depth(),
    min_n(),
    sample_size = sample_prop(),
    finalize(mtry(), fair_train),
    size = 10)
  
  # Fair Workflow
  fair_wf <- workflow()%>%
    add_recipe(fair_rec)%>%
    add_model(fair_spec)
  
  # Fair Folds
  set.seed(957)
  fair_folds <- vfold_cv(fair_train, v = 5, strata = fair)
  
  # Tune Fair Model
  set.seed(805)
  fair_tune <- tune_grid(
    fair_wf,
    resamples = fair_folds,
    grid = fair_grid,
    control = control_grid(save_pred = TRUE))
  
  # Finalize Fair Workflow
  final_fair <- finalize_workflow(
    fair_wf,
    select_best(fair_tune, "roc_auc"))
  
  # Fit Workflow
  fair_fit <- 
    final_fair %>%
    last_fit(fair_split) 
  
  fair_model <- butcher::axe_env.workflow(extract_workflow(fair_fit))
  
  # Save Fair Model
  fair_mod_name <- paste0(pitch_type_name, "_", "loc_fair_mod.rds")
  readr::write_rds(fair_model, fair_mod_name)
  
  # Create Inplay Dataframe
  inplay_data <- data3%>%
    dplyr::mutate(fair = ifelse(description == "hit_into_play",
                                "fair", "foul"))%>%
    dplyr::filter(fair == "fair", type == "X",
                  !is.na(launch_speed), !is.na(launch_angle))%>%
    dplyr::mutate(la_group = ifelse(launch_angle <= 10, "ground_ball",
                                    ifelse(launch_angle > 10 & launch_angle <= 25, "line_drive",
                                           ifelse(launch_angle > 25 & launch_angle <= 45, "fly_ball",
                                                  "pop_up"))))%>%
    dplyr::mutate(ev_group = ifelse(launch_speed < 90, "weak",
                                    ifelse(launch_speed >= 90 & launch_speed < 95, "medium",
                                           ifelse(launch_speed >= 95 & launch_speed < 100, "hard",
                                                  ifelse(launch_speed >= 100 & launch_speed < 105, "very_hard",
                                                         "smoked")))))%>%
    dplyr::mutate(bb_group = ifelse(la_group == "pop_up", la_group, paste0(la_group, "_", ev_group)))%>%
    dplyr::filter(!is.na(bb_group))%>%
    dplyr::mutate(bb_group = as.factor(bb_group))
  
  # Split into Training and Testing
  set.seed(575)
  ip_split <- initial_split(ip_data, strata = bb_group)
  ip_train <- training(ip_split)
  ip_test  <- testing(ip_split)
  
  # Inplay Model Spec
  ip_spec <- boost_tree(
    trees = 100, 
    tree_depth = tune(), min_n = tune(), 
    sample_size = tune(), mtry = tune(),         
    learn_rate = 0.1) %>% 
    set_engine("xgboost", num_class = 16) %>% 
    set_mode("classification")
  
  # Inplay Recipe
  ip_rec <- recipe(bb_group ~ plate_x + plate_z + balls + strikes + 
                       + p_throws + stand, data = ip_train)%>%
    step_dummy(balls, strikes, stand, p_throws, one_hot = T)
  
  # Inplay Grid
  ip_grid <- grid_latin_hypercube(
    tree_depth(),
    min_n(),
    sample_size = sample_prop(),
    finalize(mtry(), ip_train),
    size = 10)
  
  # Inplay Workflow
  ip_wf <- workflow()%>%
    add_recipe(ip_rec)%>%
    add_model(ip_spec)
  
  # Inplay Folds
  set.seed(724)
  ip_folds <- vfold_cv(ip_train, v = 5, strata = bb_group)
  
  # Tune Inplay Model
  set.seed(837)
  ip_tune <- tune_grid(
    ip_wf,
    resamples = ip_folds,
    grid = ip_grid,
    control = control_grid(save_pred = TRUE))
  
  # Finalize Inplay Workflow
  final_ip <- finalize_workflow(
    ip_wf,
    select_best(ip_tune, "roc_auc"))
  
  # Fit Workflow
  ip_fit <- 
    final_ip %>%
    last_fit(ip_split) 
  
  inplay_model <- butcher::axe_env.workflow(extract_workflow(ip_fit))
  
  # Save Inplay Model
  inplay_mod_name <- paste0(pitch_type_name, "_", "loc_inplay_mod.rds")
  readr::write_rds(inplay_model, inplay_mod_name)
  
  return(list(swing_mod = swing_fit, take_mod = take_fit, 
              contact_mod = contact_fit, 
              fair_mod = fair_fit, inplay_mod = ip_fit))
  
}


# Load in data
library(readr)
library(baseballr)
data1 <- read_csv("pitcher_data_2020.csv")
data2 <- read_csv("pitcher_data_2021.csv")

d1 <- scrape_statcast_savant(start_date = "2022-04-07", 
                             end_date = "2022-04-15",
                             player_type = "pitcher") 

d2 <- scrape_statcast_savant(start_date = "2022-04-16", 
                             end_date = "2022-04-24",
                             player_type = "pitcher") 

d3 <- scrape_statcast_savant(start_date = "2022-04-25", 
                             end_date = "2022-04-30",
                             player_type = "pitcher")

d4 <- scrape_statcast_savant(start_date = "2022-05-01", 
                             end_date = "2022-05-09",
                             player_type = "pitcher")

d5 <- scrape_statcast_savant(start_date = "2022-05-10", 
                             end_date = "2022-05-18",
                             player_type = "pitcher")

d6 <- scrape_statcast_savant(start_date = "2022-05-19", 
                             end_date = "2022-05-27",
                             player_type = "pitcher")

d7 <- scrape_statcast_savant(start_date = "2022-05-28", 
                             end_date = "2022-06-04",
                             player_type = "pitcher")

d8 <- scrape_statcast_savant(start_date = "2022-06-05", 
                             end_date = "2022-06-12",
                             player_type = "pitcher")

d9 <- scrape_statcast_savant(start_date = "2022-06-13", 
                             end_date = "2022-06-20",
                             player_type = "pitcher")

d10 <- scrape_statcast_savant(start_date = "2022-06-21", 
                              end_date = "2022-06-27",
                              player_type = "pitcher")

d11 <- scrape_statcast_savant(start_date = "2022-06-28", 
                              end_date = "2022-07-04",
                              player_type = "pitcher")

data3 <- rbind(d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11)
rm(d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11)

#Combine seasons into one dataframe

full_data <- rbind(data1[,-1], data2[,-1], data3)%>%
  filter(!balls == 4,
         !strikes == 3)

#Seperate pitch types

#Four Seamers

ff_data <- full_data%>%
  filter(pitch_type == "FF")

#Sinkers

si_data <- full_data%>%
  filter(pitch_type == "SI")

#Cutters

fc_data <- full_data%>%
  filter(pitch_type == "FC")

#Changeups and Splitters

ch_data <- full_data%>%
  filter(pitch_type %in% c("CH", "FS"))

#Sliders

sl_data <- full_data%>%
  filter(pitch_type == "SL")

#Curveballs

cb_data <- full_data%>%
  filter(pitch_type %in% c("CU", "KC"))

#Train Models

#Four Seamers

ff_mods <- build_location_tree("ff", ff_data)

#Sinkers

si_mods <- build_location_tree("si", si_data)

#Cutters

fc_mods <- build_location_tree("fc", fc_data)

#Changeups and Splitters

ch_mods <- build_location_tree("ch", ch_data)

#Curveballs

cb_mods <- build_location_tree("cb", cb_data)

#Sliders

sl_mods <- build_location_tree("sl", sl_data)

#Finding Run Values

take_rv <- full_data%>%
  mutate(swing = ifelse(description %in% c("foul_tip", "swinging_strike",  
                                               "swinging_strike_blocked", 
                                               "missed_bunt",
                                               "foul", "hit_into_play", 
                                               "foul_bunt", "bunt_foul_tip"),
                            "swing", "take"))%>%
  filter(swing == "take")%>%
  mutate(outcome = ifelse(description == "called_strike", "called_strike",
                              ifelse(description %in% c("ball", "blocked_ball", 
                                                        "pitchout", "foul_pitchout"),
                                     "ball", "hit_by_pitch")))%>%
  filter(!is.na(outcome))%>%
  group_by(outcome)%>%
  summarize(mean_rv = mean(delta_run_exp, na.rm = T))%>%
  ungroup()

whiff_rv <- full_data%>%
  mutate(swing = ifelse(description %in% c("foul_tip", "swinging_strike",  
                                           "swinging_strike_blocked", 
                                           "missed_bunt",
                                           "foul", "hit_into_play", 
                                           "foul_bunt", "bunt_foul_tip"),
                        "swing", "take"),
         contact = ifelse(description %in% c("hit_into_play", "foul",
                                   "foul_pitchout"),
                          "contact", "swinging_strike"))%>%
  filter(swing == "swing", contact == "swinging_strike")%>%
  filter(!is.na(contact))%>%
  group_by(contact)%>%
  summarize(mean_rv = mean(delta_run_exp, na.rm = T))%>%
  ungroup()%>%
  rename(outcome = contact)

foul_rv <- full_data%>%
  mutate(swing = ifelse(description %in% c("foul_tip", "swinging_strike",  
                                           "swinging_strike_blocked", 
                                           "missed_bunt",
                                           "foul", "hit_into_play", 
                                           "foul_bunt", "bunt_foul_tip"),
                        "swing", "take"),
         foul = ifelse(description %in% c("foul", "foul_pitchout"),
                          "foul", "fair"))%>%
  filter(swing == "swing", foul == "foul")%>%
  filter(!is.na(foul))%>%
  group_by(foul)%>%
  summarize(mean_rv = mean(delta_run_exp, na.rm = T))%>%
  ungroup()%>%
  rename(outcome = foul)

inplay_rv <- full_data%>%
  dplyr::mutate(fair = ifelse(description == "hit_into_play",
                              "fair", "foul"))%>%
  dplyr::filter(fair == "fair", type == "X",
                !is.na(launch_speed), !is.na(launch_angle))%>%
  dplyr::mutate(la_group = ifelse(launch_angle <= 10, "ground_ball",
                                  ifelse(launch_angle > 10 & launch_angle <= 25, "line_drive",
                                         ifelse(launch_angle > 25 & launch_angle <= 45, "fly_ball",
                                                "pop_up"))))%>%
  dplyr::mutate(ev_group = ifelse(launch_speed < 90, "weak",
                                  ifelse(launch_speed >= 90 & launch_speed < 95, "medium",
                                         ifelse(launch_speed >= 95 & launch_speed < 100, "hard",
                                                ifelse(launch_speed >= 100 & launch_speed < 105, "very_hard",
                                                       "smoked")))))%>%
  dplyr::mutate(bb_group = ifelse(la_group == "pop_up", la_group, paste0(la_group, "_", ev_group)))%>%
  dplyr::filter(!is.na(bb_group))%>%
  dplyr::mutate(bb_group = as.factor(bb_group))%>%
  dplyr::group_by(bb_group)%>%
  dplyr::summarize(mean_rv = mean(delta_run_exp, na.rm = T))%>%
  dplyr::rename(outcome = bb_group)

rv_df <- rbind(take_rv, whiff_rv, foul_rv, inplay_rv)%>%
  arrange(desc(mean_rv))
write.csv(rv_df, "rv_df.csv") 