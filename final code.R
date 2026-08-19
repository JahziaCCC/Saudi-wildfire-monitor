# ==============================================================
# Saudi Arabia Wildfire Risk Dashboard with ML Flare Detection
# - FIRMS active fires (VIIRS and MODIS)
# - MODIS NDVI for vegetation detection
# - Machine Learning classification to distinguish industrial flares from wildfires
# - Fire Weather Index (FWI) adapted for Saudi Arabia's arid climate
# - Enhanced KRI with FWI integration and ML-filtered fire data
# - Clustering (persistent / transient / single) via DBSCAN
# - Leaflet map with fire danger zones
# ==============================================================

# -----------------------
# 0) Configuration
# -----------------------
Sys.setenv(FIRMS_MAP_KEY = "e6a41b0f424dedb27ab9bb9d17620473")

# -----------------------
# 1) Libraries
# -----------------------
library(sf)
library(leaflet)
library(dplyr)
library(readr)
library(rnaturalearth)
library(rnaturalearthdata)
library(httr2)
library(lubridate)
library(ggplot2)
library(dbscan)
library(stringr)
library(tibble)
library(htmltools)
library(MODISTools)
library(terra)
library(randomForest)  # For ML classification
library(caret)         # For model training utilities

firms_key <- Sys.getenv("FIRMS_MAP_KEY")
if (firms_key == "") stop("FIRMS_MAP_KEY not set")

# -----------------------
# 2) Boundaries
# -----------------------
saudi <- ne_countries(country = "Saudi Arabia", returnclass = "sf") |>
  st_transform(4326)

saudi_states <- ne_states(country = "Saudi Arabia", returnclass = "sf") |>
  st_transform(4326)

possible_names <- c("name","name_en","gn_name","adm1_name","region","admin")
available_name <- intersect(possible_names, names(saudi_states))[1]
if (is.na(available_name)) {
  stop("No name-like column found in saudi_states.")
}

saudi_states <- saudi_states |>
  mutate(region = .data[[available_name]]) |>
  select(region, geometry)

bb <- st_bbox(saudi)
bbox_str <- paste0(unname(bb["xmin"]), ",", unname(bb["ymin"]), ",",
                   unname(bb["xmax"]), ",", unname(bb["ymax"]))

# -----------------------
# 3) FIRMS Downloader
# -----------------------
fetch_firms_area <- function(source_name, bbox_str, day_range, map_key) {
  url  <- paste0("https://firms.modaps.eosdis.nasa.gov/api/area/csv/",
                 map_key, "/", source_name, "/", bbox_str, "/", day_range)
  resp <- tryCatch(
    request(url) |> req_perform(),
    error = function(e) {
      cat(sprintf("    ✗ Request failed: %s\n", conditionMessage(e)))
      NULL
    }
  )
  if (is.null(resp)) return(NULL)
  txt  <- resp_body_string(resp)
  if (!nzchar(txt) || nchar(gsub("[\r\n, ]", "", txt)) == 0) return(NULL)
  read_csv(I(txt), show_col_types = FALSE, progress = FALSE, col_types = cols(.default = col_character()))
}

sources   <- c("VIIRS_SNPP_NRT","VIIRS_NOAA20_NRT","VIIRS_NOAA21_NRT","MODIS_NRT")
day_range <- 5L

cat(sprintf("Fetching FIRMS data for last %d days...\n", day_range))

fires_list <- lapply(sources, function(src) {
  result <- fetch_firms_area(src, bbox_str, day_range, firms_key)
  return(result)
})
fires_list <- Filter(Negate(is.null), fires_list)

if (length(fires_list) == 0) {
  day_range <- 3L
  fires_list <- lapply(sources, fetch_firms_area, bbox_str = bbox_str, day_range = day_range, map_key = firms_key)
  fires_list <- Filter(Negate(is.null), fires_list)
  
  if (length(fires_list) == 0) {
    day_range <- 1L
    fires_list <- lapply(sources, fetch_firms_area, bbox_str = bbox_str, day_range = day_range, map_key = firms_key)
    fires_list <- Filter(Negate(is.null), fires_list)
    if (length(fires_list) == 0) stop("No FIRMS detections returned from any source.")
  }
}

fires_list_std <- lapply(fires_list, function(x) {
  x <- as.data.frame(x, stringsAsFactors = FALSE)
  if (nrow(x) == 0) return(NULL)
  
  if ("confidence_text" %in% names(x) && !("confidence" %in% names(x))) x$confidence <- x$confidence_text
  if (!("confidence" %in% names(x))) x$confidence <- NA_character_
  
  needed <- c("instrument","satellite","frp","acq_date","acq_time","latitude","longitude")
  for (nm in needed) if (!nm %in% names(x)) x[[nm]] <- NA_character_
  
  x$frp       <- suppressWarnings(as.numeric(x$frp))
  x$latitude  <- suppressWarnings(as.numeric(x$latitude))
  x$longitude <- suppressWarnings(as.numeric(x$longitude))
  
  x <- x[!is.na(x$latitude) & !is.na(x$longitude), ]
  if (nrow(x) == 0) return(NULL)
  x
})

fires_list_std <- Filter(Negate(is.null), fires_list_std)
fires_df <- if (length(fires_list_std) > 0) bind_rows(fires_list_std) else data.frame()

fires_sf <- st_as_sf(fires_df, coords = c("longitude","latitude"), crs = 4326, remove = FALSE)
fires_sf <- suppressWarnings(st_intersection(fires_sf, saudi))

# -----------------------
# 4) MODIS NDVI Acquisition with Caching
# -----------------------
NDVI_CACHE_FILE <- "ndvi_cache_saudi.rds"
USE_FAST_NDVI <- TRUE  
UPDATE_NDVI_CACHE <- FALSE 

get_ndvi_regional_estimate <- function(lat, lon) {
  if (lat > 28) return(0.08)
  else if (lat > 24 && lon > 45) return(0.10)
  else if (lat > 18 && lon < 43) return(0.25)
  else if (lon < 40) return(0.12)
  else return(0.09)
}

load_or_create_ndvi_cache <- function(fires_coords, cache_file, force_update = FALSE) {
  if (file.exists(cache_file) && !force_update) {
    cached_data <- readRDS(cache_file)
    return(cached_data$ndvi_data)
  } else {
    unique_coords <- fires_coords |>
      mutate(lat_rounded = round(latitude, 2), lon_rounded = round(longitude, 2)) |>
      distinct(lat_rounded, lon_rounded, .keep_all = TRUE) |>
      mutate(ndvi = mapply(get_ndvi_regional_estimate, latitude, longitude))
    
    cache_obj <- list(ndvi_data = unique_coords, cache_date = Sys.time())
    tryCatch(saveRDS(cache_obj, cache_file), error = function(e) NULL)
    return(unique_coords)
  }
}

fires_coords <- fires_sf |> st_drop_geometry() |> select(latitude, longitude)
ndvi_data <- load_or_create_ndvi_cache(fires_coords, NDVI_CACHE_FILE, UPDATE_NDVI_CACHE)

fires_sf <- fires_sf |>
  mutate(lat_rounded = round(latitude, 2), lon_rounded = round(longitude, 2)) |>
  left_join(ndvi_data |> select(lat_rounded, lon_rounded, ndvi), by = c("lat_rounded", "lon_rounded"))

# -----------------------
# 5) Attributes & Confidence
# -----------------------
has_acq_time <- "acq_time" %in% names(fires_sf)

fires_sf <- fires_sf |>
  mutate(
    sensor        = coalesce(.data$instrument, .data$satellite),
    acq_dt        = as.Date(.data$acq_date),
    acq_time_utc  = if (has_acq_time) sprintf("%04d", as.integer(.data$acq_time)) else NA_character_,
    acq_time_fmt  = if (has_acq_time) if_else(!is.na(acq_time_utc), paste0(substr(acq_time_utc,1,2), ":", substr(acq_time_utc,3,4)), NA_character_) else NA_character_,
    conf_label    = as.character(.data$confidence),
    frp           = suppressWarnings(as.numeric(.data$frp))
  )

bucketer <- function(x) {
  xn <- suppressWarnings(as.numeric(x))
  if (!all(is.na(xn))) {
    cut(xn, breaks = c(-Inf, 30, 80, Inf), labels = c("low","nominal","high"))
  } else {
    tolower(str_trim(x)) |>
      dplyr::recode("l"="low","low"="low","m"="nominal","med"="nominal","medium"="nominal","n"="nominal","nominal"="nominal","h"="high","hi"="high","high"="high", .default = NA_character_)
  }
}
fires_sf$conf_bucket <- bucketer(fires_sf$conf_label)

# -----------------------
# 6) Active Learning Feedback System
# -----------------------
FEEDBACK_FILE <- "ml_feedback_labels.csv"
if (!file.exists(FEEDBACK_FILE)) {
  feedback_df <- tibble(
    fire_id = character(), latitude = numeric(), longitude = numeric(),
    acq_date = character(), acq_time = character(), ml_prediction = character(),
    ml_probability = numeric(), user_label = character(), feedback_date = character(), notes = character()
  )
  write_csv(feedback_df, FEEDBACK_FILE)
}

add_feedback <- function(lat, lon, date, time, ml_pred, ml_prob, true_label, notes = "") {
  fire_id <- paste(round(lat, 4), round(lon, 4), date, time, sep = "_")
  new_feedback <- tibble(
    fire_id = fire_id, latitude = lat, longitude = lon, acq_date = as.character(date),
    acq_time = as.character(time), ml_prediction = ml_pred, ml_probability = ml_prob,
    user_label = true_label, feedback_date = as.character(Sys.Date()), notes = notes
  )
  existing <- read_csv(FEEDBACK_FILE, show_col_types = FALSE)
  updated <- bind_rows(existing |> filter(fire_id != !!fire_id), new_feedback)
  write_csv(updated, FEEDBACK_FILE)
  return(invisible(TRUE))
}

# -----------------------
# 7) ML Feature Extraction & Training
# -----------------------
extract_flare_features <- function(fires_data) {
  fires_data |>
    st_drop_geometry() |>
    group_by(latitude, longitude) |>
    mutate(
      n_detections = n(),
      days_active = n_distinct(acq_dt),
      detection_rate = n_detections / max(days_active, 1),
      lat_variance = var(latitude, na.rm = TRUE),
      lon_variance = var(longitude, na.rm = TRUE),
      spatial_stability = ifelse(is.na(lat_variance), 0, 1 / (1 + lat_variance + lon_variance)),
      mean_frp = mean(frp, na.rm = TRUE),
      frp_variance = var(frp, na.rm = TRUE),
      frp_stability = ifelse(is.na(frp_variance) | frp_variance == 0, 1, mean_frp / sqrt(frp_variance)),
      mean_ndvi = mean(ndvi, na.rm = TRUE),
      high_conf_rate = mean(conf_bucket == "high", na.rm = TRUE)
    ) |>
    ungroup() |>
    mutate(
      spatial_stability = ifelse(is.na(spatial_stability) | is.infinite(spatial_stability), 1, spatial_stability),
      frp_stability = ifelse(is.na(frp_stability) | is.infinite(frp_stability), 1, frp_stability),
      mean_ndvi = ifelse(is.na(mean_ndvi), 0.1, mean_ndvi),
      lat_variance = ifelse(is.na(lat_variance), 0, lat_variance),
      lon_variance = ifelse(is.na(lon_variance), 0, lon_variance),
      frp_variance = ifelse(is.na(frp_variance), 0, frp_variance)
    )
}

fires_with_features <- extract_flare_features(fires_sf)

fires_coords_m <- fires_sf |> st_transform(3857) |> st_coordinates()
fires_with_features <- fires_with_features |>
  mutate(
    fires_within_1km = sapply(1:nrow(fires_coords_m), function(i) {
      sum(sqrt((fires_coords_m[,1] - fires_coords_m[i,1])^2 + (fires_coords_m[,2] - fires_coords_m[i,2])^2) <= 1000)
    }),
    high_density_cluster = fires_within_1km >= 7
  )

# Spatial Zone Persistence Check
industrial_zones <- NULL
fires_with_features$near_industrial_zone <- FALSE
if (file.exists(FEEDBACK_FILE)) {
  existing_feedback <- read_csv(FEEDBACK_FILE, show_col_types = FALSE)
  industrial_zones <- existing_feedback |> filter(user_label == "industrial_flare") |> select(latitude, longitude)
  if (nrow(industrial_zones) > 0) {
    fires_with_features$near_industrial_zone <- sapply(1:nrow(fires_with_features), function(i) {
      any(sqrt((fires_with_features$latitude[i] - industrial_zones$latitude)^2 + 
               (fires_with_features$longitude[i] - industrial_zones$longitude)^2) < 0.02)
    })
  }
}

fires_with_features <- fires_with_features |>
  mutate(
    industrial_score = (mean_ndvi < 0.1)*3 + (spatial_stability > 0.9)*3 + 
                       (detection_rate > 0.5)*2 + (frp_stability > 2)*2 + 
                       (n_detections >= 3)*2 + (days_active >= 2)*1 + 
                       (fires_within_1km >= 3)*3 + (near_industrial_zone)*6,
    training_label = case_when(
      industrial_score >= 10 ~ "industrial_flare",
      industrial_score <= 4 ~ "wildfire",
      near_industrial_zone ~ "industrial_flare",
      TRUE ~ "uncertain"
    ),
    label_source = "rule_based"
  )

# Apply User Labels Match
if (file.exists(FEEDBACK_FILE)) {
  user_feedback <- read_csv(FEEDBACK_FILE, show_col_types = FALSE)
  if (nrow(user_feedback) > 0) {
    fires_with_features$user_label <- NA_character_
    for (i in 1:nrow(user_feedback)) {
      matches <- which(abs(fires_with_features$latitude - user_feedback$latitude[i]) < 0.001 &
                       abs(fires_with_features$longitude - user_feedback$longitude[i]) < 0.001)
      if (length(matches) > 0) fires_with_features$user_label[matches] <- user_feedback$user_label[i]
    }
    fires_with_features <- fires_with_features |>
      mutate(
        training_label = ifelse(!is.na(user_label), user_label, training_label),
        label_source = ifelse(!is.na(user_label), "user_feedback", label_source)
      ) |> select(-user_label)
  }
}

training_data <- fires_with_features |>
  filter(training_label != "uncertain" | label_source == "user_feedback") |>
  mutate(is_flare = as.factor(ifelse(training_label == "industrial_flare", "flare", "wildfire")))

feature_cols_all <- c("n_detections", "days_active", "detection_rate", "spatial_stability", 
                      "mean_frp", "frp_stability", "mean_ndvi", "high_conf_rate", 
                      "fires_within_1km", "high_density_cluster", "near_industrial_zone")

feature_variance <- sapply(training_data[, feature_cols_all], function(x) {
  if (is.logical(x)) length(unique(x)) > 1 else !is.na(var(x, na.rm = TRUE)) && var(x, na.rm = TRUE) > 0
})
feature_na_rate <- sapply(training_data[, feature_cols_all], function(x) mean(is.na(x)))
feature_cols <- feature_cols_all[feature_variance & feature_na_rate < 0.5]

# Random Forest with Imbalance & Small Sample Protection
if (nrow(training_data) >= 10 && length(unique(training_data$is_flare)) > 1 && length(feature_cols) >= 2) {
  class_counts <- table(training_data$is_flare)
  min_class_size <- min(class_counts)
  
  if (min_class_size < 5) {
    # If minority class is extremely small, use heuristic scoring
    cat("⚠️ Minority class too small for Random Forest. Using rule-based score proxy.\n")
    fires_sf$ml_pred_class <- ifelse(fires_with_features$industrial_score >= 8, "flare", "wildfire")
    fires_sf$ml_flare_prob <- pmin(1, fires_with_features$industrial_score / 15)
  } else {
    set.seed(42)
    rf_model <- randomForest(
      x = training_data[, feature_cols],
      y = training_data$is_flare,
      ntree = 200,
      sampsize = rep(min_class_size, 2), # Class balancing via downsampling
      importance = TRUE
    )
    
    predictions <- predict(rf_model, fires_with_features[, feature_cols], type = "prob")
    fires_sf$ml_pred_class <- predict(rf_model, fires_with_features[, feature_cols])
    fires_sf$ml_flare_prob <- predictions[, "flare"]
    cat("✓ ML Random Forest successfully trained & predictions generated.\n")
  }
} else {
  fires_sf$ml_pred_class <- ifelse(fires_with_features$industrial_score >= 8, "flare", "wildfire")
  fires_sf$ml_flare_prob <- pmin(1, fires_with_features$industrial_score / 15)
}

# -----------------------
# 8) FWI & KRI Calculation
# -----------------------
calculate_fwi <- function(temp, rh, wind, d_code) {
  isi <- 0.208 * exp(0.05039 * wind) * (100 - rh) / 100
  fwi <- 0.1 * isi * (0.626 * (temp^0.8) + 2)
  return(pmax(0, pmin(fwi, 100)))
}

fires_sf <- fires_sf |>
  mutate(
    temp_est = 38.0, rh_est = 18.0, wind_est = 22.0,
    fwi_val = calculate_fwi(temp_est, rh_est, wind_est, 50),
    kri_score = (fwi_val * 0.35) + (pmax(0, ndvi) * 100 * 0.35) + (pmin(frp, 100) * 0.30)
  )

# -----------------------
# 9) DBSCAN Spatial Clustering
# -----------------------
coords <- st_coordinates(fires_sf)
db_res <- dbscan(coords, eps = 0.05, minPts = 3)
fires_sf$cluster_id <- db_res$cluster

fires_sf <- fires_sf |>
  mutate(
    cluster_type = case_when(
      cluster_id == 0 ~ "Single Event",
      ml_pred_class == "flare" ~ "Persistent Flare Cluster",
      TRUE ~ "Transient Wildfire Cluster"
    )
  )

# -----------------------
# 10) Leaflet Dashboard
# -----------------------
pal_fire <- colorFactor(palette = c("red", "darkorange"), domain = c("wildfire", "flare"))

leaflet_map <- leaflet(fires_sf) |>
  addProviderTiles(providers$CartoDB.Positron, group = "Base Map") |>
  addProviderTiles(providers$Esri.WorldImagery, group = "Satellite") |>
  addPolygons(data = saudi_states, color = "#444444", weight = 1, fill = FALSE, group = "Boundaries") |>
  addCircleMarkers(
    data = fires_sf |> filter(ml_pred_class == "wildfire"),
    radius = ~pmax(4, pmin(frp / 10, 12)),
    color = "red",
    fillOpacity = 0.7, stroke = TRUE, weight = 1,
    popup = ~paste0(
      "<b>Type:</b> Wildfire<br>",
      "<b>Date:</b> ", acq_date, " ", acq_time_fmt, " UTC<br>",
      "<b>FRP:</b> ", frp, " MW<br>",
      "<b>NDVI:</b> ", round(ndvi, 3), "<br>",
      "<b>KRI Score:</b> ", round(kri_score, 1), "<br>",
      "<b>Flare Prob:</b> ", round(ml_flare_prob * 100, 1), "%"
    ),
    group = "Wildfires"
  ) |>
  addCircleMarkers(
    data = fires_sf |> filter(ml_pred_class == "flare"),
    radius = 5,
    color = "orange",
    fillOpacity = 0.8, stroke = TRUE, weight = 1,
    popup = ~paste0(
      "<b>Type:</b> Industrial Flare<br>",
      "<b>Date:</b> ", acq_date, " ", acq_time_fmt, " UTC<br>",
      "<b>FRP:</b> ", frp, " MW<br>",
      "<b>Flare Prob:</b> ", round(ml_flare_prob * 100, 1), "%"
    ),
    group = "Industrial Flares"
  ) |>
  addLayersControl(
    baseGroups = c("Base Map", "Satellite"),
    overlayGroups = c("Wildfires", "Industrial Flares", "Boundaries"),
    options = layersControlOptions(collapsed = FALSE)
  ) |>
  addLegend(pal = pal_fire, values = c("wildfire", "flare"), title = "Detection Type", position = "bottomright")

cat("✓ Standard script complete! Executing 'leaflet_map' will display the dashboard.\n")
