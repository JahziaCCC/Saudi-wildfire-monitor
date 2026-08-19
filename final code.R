# ==============================================================
# Saudi Arabia Wildfire Risk Dashboard with ML Flare Detection
# - FIRMS active fires (VIIRS and MODIS)
# - MODIS NDVI for vegetation detection
# - Machine Learning classification to distinguish industrial flares from wildfires
# - Fire Weather Index (FWI) adapted for Saudi Arabia's arid climate
# - Enhanced KRI with FWI integration and ML-filtered fire data
# - Clustering (persistent / transient / single)
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
  stop("No name-like column found in saudi_states. Columns: ",
       paste(names(saudi_states), collapse = ", "))
}

saudi_states <- saudi_states |>
  mutate(region = .data[[available_name]]) |>
  select(region, geometry)

bb <- st_bbox(saudi)
bbox_str <- paste0(unname(bb["xmin"]), ",", unname(bb["ymin"]), ",",
                   unname(bb["xmax"]), ",", unname(bb["ymax"]))

# -----------------------
# 3) FIRMS downloader
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
  read_csv(I(txt),
           show_col_types = FALSE,
           progress = FALSE,
           col_types = cols(.default = col_character()))
}

sources   <- c("VIIRS_SNPP_NRT","VIIRS_NOAA20_NRT","VIIRS_NOAA21_NRT","MODIS_NRT")
day_range <- 5L  # FIRMS area/csv API caps day_range at 5 days per request

cat(sprintf("Fetching FIRMS data for last %d days...\n", day_range))

fires_list <- lapply(sources, function(src) {
  cat(sprintf("  Trying %s...\n", src))
  result <- fetch_firms_area(src, bbox_str, day_range, firms_key)
  if (!is.null(result) && nrow(result) > 0) {
    cat(sprintf("    ✓ Got %d detections\n", nrow(result)))
  } else {
    cat(sprintf("    ✗ No data\n"))
  }
  return(result)
})

fires_list <- Filter(Negate(is.null), fires_list)

if (length(fires_list) == 0) {
  cat("\n⚠️  No data from any source for 5 days. Trying with 3 days...\n\n")
  day_range <- 3L
  
  fires_list <- lapply(sources, function(src) {
    cat(sprintf("  Trying %s (3 days)...\n", src))
    result <- fetch_firms_area(src, bbox_str, day_range, firms_key)
    if (!is.null(result) && nrow(result) > 0) {
      cat(sprintf("    ✓ Got %d detections\n", nrow(result)))
    }
    return(result)
  })
  
  fires_list <- Filter(Negate(is.null), fires_list)
  
  if (length(fires_list) == 0) {
    cat("\n⚠️  No data for 3 days. Trying with 1 day...\n\n")
    day_range <- 1L

    fires_list <- lapply(sources, fetch_firms_area,
                         bbox_str = bbox_str, day_range = day_range, map_key = firms_key)
    fires_list <- Filter(Negate(is.null), fires_list)
    
    if (length(fires_list) == 0) {
      stop("No FIRMS detections returned from any source. Check your API key and internet connection.")
    }
  }
}

fires_list_std <- lapply(fires_list, function(x) {
  x <- as.data.frame(x, stringsAsFactors = FALSE)
  
  # Skip if empty
  if (nrow(x) == 0) return(NULL)  # Return NULL instead of empty df
  
  # Print column names for debugging
  cat(sprintf("    Columns in data: %s\n", paste(head(names(x), 10), collapse=", ")))
  
  if ("confidence_text" %in% names(x) && !("confidence" %in% names(x))) {
    x$confidence <- x$confidence_text
  }
  if (!("confidence" %in% names(x))) x$confidence <- NA_character_
  x$confidence <- as.character(x$confidence)
  
  needed <- c("instrument","satellite","frp","acq_date","acq_time","latitude","longitude")
  for (nm in needed) if (!nm %in% names(x)) x[[nm]] <- NA_character_
  
  x$frp       <- suppressWarnings(as.numeric(x$frp))
  x$latitude  <- suppressWarnings(as.numeric(x$latitude))
  x$longitude <- suppressWarnings(as.numeric(x$longitude))
  
  # Filter out rows with invalid coordinates
  x <- x[!is.na(x$latitude) & !is.na(x$longitude), ]
  
  cat(sprintf("    After standardization: %d rows remain\n", nrow(x)))
  
  if (nrow(x) == 0) return(NULL)
  
  x
})

# Remove NULL entries
fires_list_std <- Filter(Negate(is.null), fires_list_std)

fires_df <- if (length(fires_list_std) > 0) bind_rows(fires_list_std) else data.frame()

cat(sprintf("After standardization: %d rows\n", nrow(fires_df)))

if (nrow(fires_df) == 0) {
  cat("\n⚠️  Data downloaded but became empty after standardization.\n")
  cat("This usually means:\n")
  cat("  1. No fires in Saudi Arabia bounding box for this period\n")
  cat("  2. All data was filtered out during processing\n")
  cat("  3. Temporary NASA FIRMS issue\n\n")
  
  if (day_range > 1L) {
    cat("Trying with shorter period (1 day)...\n")
    day_range <- 1L

    fires_list <- lapply(sources, function(src) {
      cat(sprintf("  Trying %s (1 day)...\n", src))
      result <- fetch_firms_area(src, bbox_str, day_range, firms_key)
      if (!is.null(result) && nrow(result) > 0) {
        cat(sprintf("    ✓ Got %d detections\n", nrow(result)))
      }
      return(result)
    })
    
    fires_list <- Filter(Negate(is.null), fires_list)
    
    if (length(fires_list) > 0) {
      fires_list_std <- lapply(fires_list, function(x) {
        x <- as.data.frame(x, stringsAsFactors = FALSE)
        if (nrow(x) == 0) return(NULL)
        
        cat(sprintf("    Retry - Columns in data: %s\n", paste(head(names(x), 10), collapse=", ")))
        
        if ("confidence_text" %in% names(x) && !("confidence" %in% names(x))) {
          x$confidence <- x$confidence_text
        }
        if (!("confidence" %in% names(x))) x$confidence <- NA_character_
        x$confidence <- as.character(x$confidence)
        
        needed <- c("instrument","satellite","frp","acq_date","acq_time","latitude","longitude")
        for (nm in needed) if (!nm %in% names(x)) x[[nm]] <- NA_character_
        
        x$frp       <- suppressWarnings(as.numeric(x$frp))
        x$latitude  <- suppressWarnings(as.numeric(x$latitude))
        x$longitude <- suppressWarnings(as.numeric(x$longitude))
        
        # Filter out rows with invalid coordinates
        x <- x[!is.na(x$latitude) & !is.na(x$longitude), ]
        
        cat(sprintf("    Retry - After standardization: %d rows remain\n", nrow(x)))
        
        if (nrow(x) == 0) return(NULL)
        
        x
      })
      
      fires_list_std <- Filter(Negate(is.null), fires_list_std)
      
      fires_df <- if (length(fires_list_std) > 0) bind_rows(fires_list_std) else data.frame()
    }
  }
  
  if (nrow(fires_df) == 0) {
    stop("No FIRMS detections after all attempts. The area may have no fires currently, or there's an API issue.")
  }
}

cat(sprintf("\n✓ Successfully retrieved data for last %d days\n", day_range))
cat(sprintf("✓ Total fire detections: %d\n\n", nrow(fires_df)))

fires_sf <- st_as_sf(fires_df, coords = c("longitude","latitude"), crs = 4326, remove = FALSE)
fires_sf <- suppressWarnings(st_intersection(fires_sf, saudi))

# -----------------------
# 4) MODIS NDVI Acquisition with Caching
# -----------------------
cat("Fetching MODIS NDVI data...\n")

# NDVI Cache Configuration
NDVI_CACHE_FILE <- "ndvi_cache_saudi.rds"  # Save in current working directory
USE_FAST_NDVI <- TRUE  # Set to TRUE to skip MODISTools and use regional estimates
UPDATE_NDVI_CACHE <- FALSE  # Set to TRUE to force update of cached NDVI data

# OPTION 1: Try MODISTools API (slow but accurate)
# OPTION 2: If fails, use coarser regional averages (fast fallback)

get_ndvi_point <- function(lat, lon, start_date, end_date, max_retries = 3) {
  for (attempt in 1:max_retries) {
    tryCatch({
      ndvi_data <- mt_subset(
        product = "MOD13Q1",
        lat = lat,
        lon = lon,
        band = "250m_16_days_NDVI",
        start = start_date,
        end = end_date,
        km_lr = 0.5,
        km_ab = 0.5,
        site_name = paste0("fire_", round(lat, 4), "_", round(lon, 4)),
        internal = TRUE,
        progress = FALSE
      )
      
      if (is.null(ndvi_data) || nrow(ndvi_data) == 0) return(NA_real_)
      ndvi_val <- as.numeric(ndvi_data$value[1]) / 10000
      if (ndvi_val < -1 || ndvi_val > 1) return(NA_real_)
      return(ndvi_val)
    }, error = function(e) {
      if (attempt < max_retries) {
        Sys.sleep(2 * attempt)  # Exponential backoff
      } else {
        cat(sprintf("Failed to get NDVI for lat=%s, lon=%s: %s\n", lat, lon, e$message))
      }
    })
  }
  return(NA_real_)
}

# Fast fallback: Use regional NDVI estimates based on Saudi geography
get_ndvi_regional_estimate <- function(lat, lon) {
  # Saudi Arabia NDVI estimates by region (based on climatology)
  # These are conservative estimates for different ecological zones
  
  if (lat > 28) {
    # Northern region - mostly desert
    return(0.08)
  } else if (lat > 24 && lon > 45) {
    # Eastern region - coastal/desert
    return(0.10)
  } else if (lat > 18 && lon < 43) {
    # Southwestern highlands (Asir) - highest vegetation
    return(0.25)
  } else if (lon < 40) {
    # Western coastal (Tihamah)
    return(0.12)
  } else {
    # Central Najd - sparse vegetation
    return(0.09)
  }
}

get_ndvi_batch <- function(fires_coords, start_date, end_date) {
  unique_coords <- fires_coords |>
    mutate(
      lat_rounded = round(latitude, 2),
      lon_rounded = round(longitude, 2)
    ) |>
    distinct(lat_rounded, lon_rounded, .keep_all = TRUE)
  
  cat(sprintf("\n=== NDVI Batch Processing ===\n"))
  cat(sprintf("Total fire detections: %d\n", nrow(fires_coords)))
  cat(sprintf("Unique locations to fetch: %d\n", nrow(unique_coords)))
  cat(sprintf("Date range: %s to %s\n\n", start_date, end_date))
  
  if (USE_FAST_NDVI) {
    cat("Using regional NDVI estimates (FAST MODE)\n")
    ndvi_results <- unique_coords |>
      mutate(ndvi = mapply(get_ndvi_regional_estimate, latitude, longitude))
    return(ndvi_results)
  }
  
  batch_size <- 25  # Reduced batch size for better success rate
  n_batches <- ceiling(nrow(unique_coords) / batch_size)
  ndvi_results <- list()
  success_count <- 0
  
  for (i in 1:n_batches) {
    start_idx <- (i - 1) * batch_size + 1
    end_idx <- min(i * batch_size, nrow(unique_coords))
    batch <- unique_coords[start_idx:end_idx, ]
    
    cat(sprintf("Batch %d/%d: Processing locations %d-%d... ", 
                i, n_batches, start_idx, end_idx))
    
    batch_start_time <- Sys.time()
    
    batch_ndvi <- lapply(1:nrow(batch), function(j) {
      ndvi <- get_ndvi_point(
        batch$latitude[j],
        batch$longitude[j],
        start_date,
        end_date
      )
      
      # Fallback to regional estimate if API fails
      if (is.na(ndvi)) {
        ndvi <- get_ndvi_regional_estimate(batch$latitude[j], batch$longitude[j])
      } else {
        success_count <<- success_count + 1
      }
      
      tibble(
        latitude = batch$latitude[j],
        longitude = batch$longitude[j],
        lat_rounded = batch$lat_rounded[j],
        lon_rounded = batch$lon_rounded[j],
        ndvi = ndvi
      )
    })
    
    ndvi_results[[i]] <- bind_rows(batch_ndvi)
    
    batch_time <- as.numeric(difftime(Sys.time(), batch_start_time, units = "secs"))
    cat(sprintf("Done (%.1fs)\n", batch_time))
    
    # Longer delay between batches to avoid rate limiting
    if (i < n_batches) {
      cat(sprintf("Waiting 5 seconds before next batch...\n"))
      Sys.sleep(5)
    }
  }
  
  cat(sprintf("\n=== NDVI Fetch Complete ===\n"))
  cat(sprintf("Successfully retrieved: %d/%d (%.1f%%)\n", 
              success_count, nrow(unique_coords),
              100 * success_count / nrow(unique_coords)))
  
  bind_rows(ndvi_results)
}

# Load or create NDVI cache
load_or_create_ndvi_cache <- function(fires_coords, cache_file, force_update = FALSE) {
  
  # Check if cache exists and we're not forcing update
  if (file.exists(cache_file) && !force_update) {
    cat(sprintf("Loading NDVI cache from: %s\n", cache_file))
    cached_data <- readRDS(cache_file)
    
    cat(sprintf("Cache contains %d locations (last updated: %s)\n", 
                nrow(cached_data$ndvi_data),
                format(cached_data$cache_date, "%Y-%m-%d %H:%M")))
    
    # Get unique fire locations
    fire_locations <- fires_coords |>
      mutate(
        lat_rounded = round(latitude, 2),
        lon_rounded = round(longitude, 2)
      ) |>
      distinct(lat_rounded, lon_rounded)
    
    # Check coverage
    cached_locations <- cached_data$ndvi_data |>
      select(lat_rounded, lon_rounded)
    
    missing_locations <- fire_locations |>
      anti_join(cached_locations, by = c("lat_rounded", "lon_rounded"))
    
    if (nrow(missing_locations) > 0) {
      cat(sprintf("Found %d new locations not in cache\n", nrow(missing_locations)))
      cat("Fetching NDVI for new locations...\n")
      
      # Add full coordinates back from fires_coords for NDVI fetching
      missing_with_coords <- fires_coords |>
        mutate(
          lat_rounded = round(latitude, 2),
          lon_rounded = round(longitude, 2)
        ) |>
        semi_join(missing_locations, by = c("lat_rounded", "lon_rounded")) |>
        distinct(lat_rounded, lon_rounded, .keep_all = TRUE)
      
      # Fetch NDVI for missing locations
      new_ndvi <- get_ndvi_batch(
        missing_with_coords,
        cached_data$date_start,
        cached_data$date_end
      )
      
      # Merge with existing cache
      updated_ndvi <- bind_rows(cached_data$ndvi_data, new_ndvi) |>
        distinct(lat_rounded, lon_rounded, .keep_all = TRUE)
      
      # Save updated cache
      updated_cache <- list(
        ndvi_data = updated_ndvi,
        date_start = cached_data$date_start,
        date_end = cached_data$date_end,
        cache_date = Sys.time()
      )
      
      saveRDS(updated_cache, cache_file)
      cat(sprintf("Cache updated with %d total locations\n", nrow(updated_ndvi)))
      
      return(updated_ndvi)
    } else {
      cat("All fire locations found in cache!\n")
      return(cached_data$ndvi_data)
    }
    
  } else {
    # Create new cache
    if (force_update) {
      cat("Force update enabled - rebuilding NDVI cache...\n")
    } else {
      cat("No cache found - creating new NDVI cache...\n")
    }
    
    # MODIS data has ~8 day processing delay
    date_end   <- Sys.Date() - 10
    date_start <- date_end - 30
    
    cat(sprintf("Using NDVI date range: %s to %s\n", 
                format(date_start, "%Y-%m-%d"), 
                format(date_end, "%Y-%m-%d")))
    
    # Get unique locations
    unique_coords <- fires_coords |>
      mutate(
        lat_rounded = round(latitude, 2),
        lon_rounded = round(longitude, 2)
      ) |>
      distinct(lat_rounded, lon_rounded, .keep_all = TRUE)
    
    # Fetch NDVI
    ndvi_data <- get_ndvi_batch(
      unique_coords,
      format(date_start, "%Y-%m-%d"),
      format(date_end, "%Y-%m-%d")
    )
    
    # Save to cache
    cache_obj <- list(
      ndvi_data = ndvi_data,
      date_start = format(date_start, "%Y-%m-%d"),
      date_end = format(date_end, "%Y-%m-%d"),
      cache_date = Sys.time()
    )
    
    # Try to save with error handling
    tryCatch({
      saveRDS(cache_obj, cache_file)
      cat(sprintf("NDVI cache saved to: %s\n", cache_file))
    }, error = function(e) {
      # Fallback to temp directory
      cache_file_alt <- file.path(tempdir(), "ndvi_cache_saudi.rds")
      saveRDS(cache_obj, cache_file_alt)
      cat(sprintf("⚠️  Could not save to %s\n", cache_file))
      cat(sprintf("Cache saved to temp location: %s\n", cache_file_alt))
      cat("Copy this file to your working directory for persistence\n")
    })
    
    cat(sprintf("Cache contains %d unique locations\n", nrow(ndvi_data)))
    
    return(ndvi_data)
  }
}

# Main NDVI acquisition logic
fires_coords <- fires_sf |>
  st_drop_geometry() |>
  select(latitude, longitude)

ndvi_data <- load_or_create_ndvi_cache(
  fires_coords, 
  NDVI_CACHE_FILE,
  force_update = UPDATE_NDVI_CACHE
)

fires_sf <- fires_sf |>
  mutate(
    lat_rounded = round(latitude, 2),
    lon_rounded = round(longitude, 2)
  ) |>
  left_join(
    ndvi_data |> select(lat_rounded, lon_rounded, ndvi),
    by = c("lat_rounded", "lon_rounded")
  )

cat(sprintf("NDVI data acquired for %d/%d fire detections\n", 
            sum(!is.na(fires_sf$ndvi)), nrow(fires_sf)))

# -----------------------
# 5) Attributes & confidence (needed for ML features)
# -----------------------
has_acq_time <- "acq_time" %in% names(fires_sf)

fires_sf <- fires_sf |>
  mutate(
    sensor        = coalesce(.data$instrument, .data$satellite),
    acq_dt        = as.Date(.data$acq_date),
    acq_time_utc  = if (has_acq_time) sprintf("%04d", as.integer(.data$acq_time)) else NA_character_,
    acq_time_fmt  = if (has_acq_time)
      if_else(!is.na(acq_time_utc),
              paste0(substr(acq_time_utc,1,2), ":", substr(acq_time_utc,3,4)),
              NA_character_)
    else NA_character_,
    conf_label    = as.character(.data$confidence),
    frp           = suppressWarnings(as.numeric(.data$frp))
  )

bucketer <- function(x) {
  xn <- suppressWarnings(as.numeric(x))
  if (!all(is.na(xn))) {
    cut(xn, breaks = c(-Inf, 30, 80, Inf),
        labels = c("low","nominal","high"))
  } else {
    tolower(str_trim(x)) |>
      dplyr::recode(
        "l"="low","low"="low",
        "m"="nominal","med"="nominal","medium"="nominal","n"="nominal","nominal"="nominal",
        "h"="high","hi"="high","high"="high",
        .default = NA_character_
      )
  }
}

fires_sf$conf_bucket <- bucketer(fires_sf$conf_label)

# -----------------------
# 6) Feedback System for Active Learning
# -----------------------
cat("\n=== Initializing Feedback System ===\n")

# Feedback storage configuration
FEEDBACK_FILE <- "ml_feedback_labels.csv"

# Initialize feedback file if it doesn't exist
if (!file.exists(FEEDBACK_FILE)) {
  feedback_df <- tibble(
    fire_id = character(),
    latitude = numeric(),
    longitude = numeric(),
    acq_date = character(),
    acq_time = character(),
    ml_prediction = character(),
    ml_probability = numeric(),
    user_label = character(),
    feedback_date = character(),
    notes = character()
  )
  write_csv(feedback_df, FEEDBACK_FILE)
  cat(sprintf("✓ Created feedback file: %s\n", FEEDBACK_FILE))
} else {
  feedback_df <- read_csv(FEEDBACK_FILE, show_col_types = FALSE)
  cat(sprintf("✓ Loaded %d existing feedback entries\n", nrow(feedback_df)))
}

# Function to add feedback
add_feedback <- function(lat, lon, date, time, ml_pred, ml_prob, true_label, notes = "") {
  fire_id <- paste(round(lat, 4), round(lon, 4), date, time, sep = "_")
  
  new_feedback <- tibble(
    fire_id = fire_id,
    latitude = lat,
    longitude = lon,
    acq_date = as.character(date),
    acq_time = as.character(time),
    ml_prediction = ml_pred,
    ml_probability = ml_prob,
    user_label = true_label,
    feedback_date = as.character(Sys.Date()),
    notes = notes
  )
  
  # Load existing feedback
  if (file.exists(FEEDBACK_FILE)) {
    existing <- read_csv(FEEDBACK_FILE, show_col_types = FALSE)
    # Remove duplicate if exists
    existing <- existing |> filter(fire_id != !!fire_id)
    updated <- bind_rows(existing, new_feedback)
  } else {
    updated <- new_feedback
  }
  
  write_csv(updated, FEEDBACK_FILE)
  cat(sprintf("✓ Feedback added for fire at (%.4f, %.4f)\n", lat, lon))
  return(invisible(TRUE))
}

cat("\n💡 To add feedback after visual inspection, use:\n")
cat("   add_feedback(lat, lon, date, time, ml_pred, ml_prob, true_label, notes)\n")
cat("   Example: add_feedback(26.123, 50.456, '2024-11-15', '1430',\n")
cat("                         'industrial_flare', 0.95, 'wildfire', 'Actually vegetation')\n\n")

# -----------------------
# 7) ML-based Industrial Flare Detection with Active Learning
# -----------------------
cat("\n=== Building ML Model for Flare Detection (with feedback) ===\n")

# Extract features for ML classification
extract_flare_features <- function(fires_data) {
  fires_data |>
    st_drop_geometry() |>
    group_by(latitude, longitude) |>
    mutate(
      # Temporal features
      n_detections = n(),
      days_active = n_distinct(acq_dt),
      detection_rate = n_detections / max(days_active, 1),
      
      # Spatial persistence (same exact location)
      lat_variance = var(latitude, na.rm = TRUE),
      lon_variance = var(longitude, na.rm = TRUE),
      spatial_stability = ifelse(is.na(lat_variance), 0, 
                                 1 / (1 + lat_variance + lon_variance)),
      
      # FRP characteristics
      mean_frp = mean(frp, na.rm = TRUE),
      max_frp = max(frp, na.rm = TRUE),
      min_frp = min(frp, na.rm = TRUE),
      frp_variance = var(frp, na.rm = TRUE),
      frp_stability = ifelse(is.na(frp_variance) | frp_variance == 0, 1,
                            mean_frp / sqrt(frp_variance)),
      
      # Vegetation features
      mean_ndvi = mean(ndvi, na.rm = TRUE),
      
      # Confidence patterns
      high_conf_rate = mean(conf_bucket == "high", na.rm = TRUE)
    ) |>
    ungroup() |>
    mutate(
      # Replace NA/Inf with sensible defaults
      spatial_stability = ifelse(is.na(spatial_stability) | is.infinite(spatial_stability), 
                                1, spatial_stability),
      frp_stability = ifelse(is.na(frp_stability) | is.infinite(frp_stability), 
                            1, frp_stability),
      mean_ndvi = ifelse(is.na(mean_ndvi), 0.1, mean_ndvi),
      lat_variance = ifelse(is.na(lat_variance), 0, lat_variance),
      lon_variance = ifelse(is.na(lon_variance), 0, lon_variance),
      frp_variance = ifelse(is.na(frp_variance), 0, frp_variance)
    )
}

# Apply feature extraction
fires_with_features <- extract_flare_features(fires_sf)

# -----------------------
# Enhanced Feature: Cluster Density Analysis (1km radius)
# -----------------------
cat("Calculating cluster density features...\n")

# Calculate local density (fires within 1km)
fires_coords_m <- fires_sf |>
  st_transform(3857) |>  # Transform to meters
  st_coordinates()

# For each fire, count neighbors within 1km
fires_with_features <- fires_with_features |>
  mutate(
    fires_within_1km = sapply(1:nrow(fires_coords_m), function(i) {
      # Calculate distances to all other fires
      dists <- sqrt((fires_coords_m[,1] - fires_coords_m[i,1])^2 + 
                    (fires_coords_m[,2] - fires_coords_m[i,2])^2)
      # Count fires within 1000m (including self)
      sum(dists <= 1000)
    }),
    # High density indicator (7+ fires in 1km as per user observation)
    high_density_cluster = fires_within_1km >= 7
  )

cat(sprintf("High-density clusters (7+ fires in 1km): %d fires\n", 
            sum(fires_with_features$high_density_cluster)))

# -----------------------
# Spatial Zone Persistence (Industrial areas stay industrial)
# -----------------------
cat("Checking for persistent industrial zones...\n")

# Load existing feedback to identify known industrial zones
industrial_zones <- NULL
if (file.exists(FEEDBACK_FILE)) {
  existing_feedback <- read_csv(FEEDBACK_FILE, show_col_types = FALSE)
  if (nrow(existing_feedback) > 0) {
    industrial_zones <- existing_feedback |>
      filter(user_label == "industrial_flare") |>
      select(latitude, longitude)
    
    if (nrow(industrial_zones) > 0) {
      cat(sprintf("Found %d known industrial flare locations from feedback\n", 
                  nrow(industrial_zones)))
      
      # For each fire, check if it's near a known industrial zone (within 2km)
      fires_with_features <- fires_with_features |>
        mutate(
          near_industrial_zone = sapply(1:n(), function(i) {
            if (nrow(industrial_zones) == 0) return(FALSE)
            
            # Get current fire's coordinates
            fire_lat <- latitude[i]
            fire_lon <- longitude[i]
            
            # Calculate distance to all known industrial zones
            dists <- sqrt((fire_lat - industrial_zones$latitude)^2 + 
                         (fire_lon - industrial_zones$longitude)^2)
            # Check if within ~2km (approximately 0.02 degrees)
            any(dists < 0.02)
          })
        )
      
      cat(sprintf("Fires near known industrial zones: %d\n", 
                  sum(fires_with_features$near_industrial_zone)))
    } else {
      fires_with_features$near_industrial_zone <- FALSE
    }
  } else {
    fires_with_features$near_industrial_zone <- FALSE
  }
} else {
  fires_with_features$near_industrial_zone <- FALSE
}

# Create training labels based on known patterns
# Industrial sources typically:
# 1. Very low NDVI (< 0.1)
# 2. High spatial stability (same location repeatedly)
# 3. High temporal persistence (detected frequently)
# 4. Consistent FRP values
# 5. Located in Eastern region (oil infrastructure)
# 6. HIGH DENSITY: Multiple fires within 1km (USER OBSERVATION - adjusted)
# 7. SPATIAL PERSISTENCE: Near previously identified industrial zones (USER OBSERVATION)

fires_with_features <- fires_with_features |>
  mutate(
    # Create training labels based on expert rules + user observations
    # Adjusted based on actual annotation patterns
    industrial_score = 
      (mean_ndvi < 0.1) * 3 +                    # Very low vegetation
      (spatial_stability > 0.9) * 3 +            # Same exact location
      (detection_rate > 0.5) * 2 +               # Frequent detections
      (frp_stability > 2) * 2 +                  # Stable FRP
      (n_detections >= 3) * 2 +                  # Multiple detections (lowered from 5)
      (days_active >= 2) * 1 +                   # Multi-day persistence (lowered from 3)
      (fires_within_1km >= 3) * 3 +              # 3+ fires in 1km (adjusted from 7)
      (near_industrial_zone) * 6,                # Near known industrial area (STRONGEST!)
    
    # Initial rule-based labels (adjusted thresholds based on user annotations)
    training_label = case_when(
      industrial_score >= 10 ~ "industrial_flare",  # Raised threshold (max now ~21)
      industrial_score <= 4 ~ "wildfire",            # Lowered threshold
      near_industrial_zone ~ "industrial_flare",     # Force industrial if near known zone
      TRUE ~ "uncertain"
    ),
    
    # Label source for tracking
    label_source = "rule_based"
  )

# Load and apply user feedback if available
user_feedback <- NULL
if (file.exists(FEEDBACK_FILE)) {
  user_feedback <- read_csv(FEEDBACK_FILE, show_col_types = FALSE)
  if (nrow(user_feedback) > 0) {
    cat(sprintf("✓ Loaded %d user-validated labels\n", nrow(user_feedback)))
    
    # Use spatial matching instead of exact fire_id (more robust)
    # Match feedback to fires based on proximity (within ~100m)
    cat("Matching feedback to fires using spatial proximity...\n")
    
    fires_with_features$user_label <- NA_character_
    
    for (i in 1:nrow(user_feedback)) {
      fb_lat <- user_feedback$latitude[i]
      fb_lon <- user_feedback$longitude[i]
      fb_label <- user_feedback$user_label[i]
      
      # Find fires within ~0.001 degrees (~100m)
      matches <- which(
        abs(fires_with_features$latitude - fb_lat) < 0.001 &
        abs(fires_with_features$longitude - fb_lon) < 0.001
      )
      
      if (length(matches) > 0) {
        # Apply label to all matching fires
        fires_with_features$user_label[matches] <- fb_label
        cat(sprintf("  Matched feedback #%d to %d fire(s)\n", i, length(matches)))
      }
    }
    
    # Override training labels with user feedback
    fires_with_features <- fires_with_features |>
      mutate(
        training_label = ifelse(!is.na(user_label), user_label, training_label),
        label_source = ifelse(!is.na(user_label), "user_feedback", label_source)
      ) |>
      select(-user_label)
    
    n_user_labels <- sum(fires_with_features$label_source == "user_feedback")
    cat(sprintf("✓ Applied %d user-validated labels to training data\n", n_user_labels))
  }
}

cat(sprintf("Training labels: Flares=%d, Wildfires=%d, Uncertain=%d\n",
            sum(fires_with_features$training_label == "industrial_flare"),
            sum(fires_with_features$training_label == "wildfire"),
            sum(fires_with_features$training_label == "uncertain")))

if (exists("label_source", where = fires_with_features)) {
  cat(sprintf("Label sources: User=%d, Rules=%d\n",
              sum(fires_with_features$label_source == "user_feedback", na.rm = TRUE),
              sum(fires_with_features$label_source == "rule_based", na.rm = TRUE)))
}

# Prepare training data (exclude uncertain cases unless user-labeled)
training_data <- fires_with_features |>
  filter(training_label != "uncertain" | label_source == "user_feedback") |>
  mutate(is_flare = training_label == "industrial_flare")

# Select features for model (including new density and zone features)
feature_cols_all <- c("n_detections", "days_active", "detection_rate",
                      "spatial_stability", "mean_frp", "frp_stability",
                      "mean_ndvi", "high_conf_rate", "lat_variance", "lon_variance",
                      "fires_within_1km", "high_density_cluster", "near_industrial_zone")

# Remove constant features and those with too many NAs
cat("\n=== Feature Selection ===\n")
feature_variance <- sapply(training_data[, feature_cols_all], function(x) {
  if (is.logical(x)) {
    # For logical, check if all TRUE or all FALSE
    return(length(unique(x)) > 1)
  } else {
    # For numeric, check variance
    var_val <- var(x, na.rm = TRUE)
    return(!is.na(var_val) && var_val > 0)
  }
})

feature_na_rate <- sapply(training_data[, feature_cols_all], function(x) {
  mean(is.na(x))
})

# Keep features with variance and < 50% missing
feature_cols <- feature_cols_all[feature_variance & feature_na_rate < 0.5]

cat("Original features:", length(feature_cols_all), "\n")
cat("Features with variance:", sum(feature_variance), "\n")
cat("Features with <50% NA:", sum(feature_na_rate < 0.5), "\n")
cat("Selected features:", length(feature_cols), "\n")
cat("Feature list:", paste(feature_cols, collapse = ", "), "\n")

if (nrow(training_data) >= 20 && length(feature_cols) >= 3) {
  # Check class balance
  class_counts <- table(training_data$is_flare)
  cat("\n=== Class Distribution ===\n")
  print(class_counts)
  
  min_class_size <- min(class_counts)
  max_class_size <- max(class_counts)
  imbalance_ratio <- max_class_size / min_class_size
  
  cat(sprintf("Class imbalance ratio: %.1f:1\n", imbalance_ratio))
  
  # Check if we have enough of minority class
  if (min_class_size < 10) {
    cat(sprintf("⚠️  Minority class has only %d samples (need 10+)\n", min_class_size))
    cat("Using rule-based classification until more annotations are available\n")
    rf_model <- NULL
  } else {
    # Train Random Forest model
    cat("Training Random Forest classifier...\n")
    
    # Prepare training matrix
    train_x <- training_data[, feature_cols]
    train_y <- as.factor(training_data$is_flare)
    
    # Handle class imbalance with stratified sampling
    if (imbalance_ratio > 3) {
      cat(sprintf("⚠️  High class imbalance (%.1f:1) - using class weights\n", imbalance_ratio))
      # Calculate class weights (inverse of frequency)
      class_weights <- 1 / as.numeric(class_counts)
      class_weights <- class_weights / sum(class_weights) * length(class_weights)
      names(class_weights) <- names(class_counts)
      
      # Create sample weights
      sample_weights <- ifelse(train_y == TRUE, 
                               class_weights["TRUE"], 
                               class_weights["FALSE"])
    } else {
      sample_weights <- NULL
    }
    
    # Train model with error handling
    rf_model <- tryCatch({
      if (!is.null(sample_weights)) {
        randomForest(
          x = train_x,
          y = train_y,
          ntree = 100,
          importance = TRUE,
          na.action = na.omit,
          classwt = class_weights,
          sampsize = rep(min(min_class_size, 100), 2)  # Balanced sampling
        )
      } else {
        randomForest(
          x = train_x,
          y = train_y,
          ntree = 100,
          importance = TRUE,
          na.action = na.omit
        )
      }
    }, error = function(e) {
      cat("❌ Random Forest training failed!\n")
      cat("Error message:", e$message, "\n")
      cat("\n=== Debugging Information ===\n")
      cat("Training data shape:", nrow(train_x), "rows x", ncol(train_x), "cols\n")
      cat("Class distribution:\n")
      print(table(train_y))
      cat("\nFeature summary:\n")
      print(summary(train_x))
      cat("\nChecking for issues:\n")
      cat("- NA values per feature:\n")
      print(colSums(is.na(train_x)))
      cat("- Infinite values per feature:\n")
      print(colSums(sapply(train_x, is.infinite)))
      cat("- Constant features:\n")
      print(sapply(train_x, function(x) length(unique(x)) == 1))
      cat("\n")
      NULL
    })
  }
  
  if (!is.null(rf_model)) {
    cat("Model trained successfully!\n")
    cat("\nFeature Importance:\n")
    print(importance(rf_model))
    
    # Predict on all data
    pred_x <- fires_with_features[, feature_cols]
    predictions <- predict(rf_model, pred_x, type = "prob")
    
    fires_with_features$flare_probability <- predictions[, "TRUE"]
    fires_with_features$ml_classification <- ifelse(
      fires_with_features$flare_probability > 0.5,
      "industrial_flare",
      "wildfire"
    )
    
    # Use ML predictions
    fires_sf$likely_flare <- fires_with_features$ml_classification == "industrial_flare"
    fires_sf$flare_probability <- fires_with_features$flare_probability
    
    cat(sprintf("\nML Classification Results:\n"))
    cat(sprintf("Industrial flares: %d (%.1f%%)\n",
                sum(fires_sf$likely_flare),
                100 * mean(fires_sf$likely_flare)))
    cat(sprintf("Wildfires: %d (%.1f%%)\n",
                sum(!fires_sf$likely_flare),
                100 * mean(!fires_sf$likely_flare)))
  } else {
    # Fallback to rule-based
    fires_sf$likely_flare <- fires_with_features$industrial_score >= 10
    fires_sf$flare_probability <- pmin(fires_with_features$industrial_score / 21, 1.0)
  }
} else {
  if (nrow(training_data) < 20) {
    cat(sprintf("⚠️  Insufficient training data (%d samples, need 20+)\n", nrow(training_data)))
  } else if (length(feature_cols) < 3) {
    cat(sprintf("⚠️  Insufficient features with variance (%d features, need 3+)\n", length(feature_cols)))
  }
  cat("Using rule-based classification\n")
  fires_sf$likely_flare <- fires_with_features$industrial_score >= 10
  fires_sf$flare_probability <- pmin(fires_with_features$industrial_score / 21, 1.0)
}

# Add NDVI classification
fires_sf <- fires_sf |>
  mutate(
    ndvi_class = case_when(
      is.na(ndvi)       ~ "unknown",
      ndvi < 0.1        ~ "barren_industrial",
      ndvi >= 0.1 & ndvi < 0.2 ~ "sparse_veg",
      ndvi >= 0.2 & ndvi < 0.4 ~ "moderate_veg",
      ndvi >= 0.4       ~ "dense_veg",
      TRUE              ~ "unknown"
    ),
    veg_fire_confidence = case_when(
      likely_flare      ~ "industrial_source",
      ndvi >= 0.2       ~ "high",
      ndvi >= 0.1       ~ "medium",
      ndvi < 0.1        ~ "low",
      TRUE              ~ "unknown"
    )
  )

cat("\n=== Fire Classification Summary ===\n")
print(fires_sf |> 
        st_drop_geometry() |> 
        count(veg_fire_confidence, ndvi_class) |> 
        arrange(desc(n)))

# -----------------------
# 8) Clustering (for BOTH vegetation fires AND industrial sources)
# -----------------------
# Cluster vegetation fires
fires_veg <- fires_sf |> filter(!likely_flare)

if (nrow(fires_veg) > 0) {
  fires_m   <- st_transform(fires_veg, 3857)
  coords_m  <- st_coordinates(fires_m)
  cl        <- dbscan(coords_m, eps = 1000, minPts = 3)
  fires_veg$cluster_id <- ifelse(cl$cluster == 0, NA_integer_, cl$cluster)
  
  cluster_persistence <- fires_veg |>
    st_drop_geometry() |>
    filter(!is.na(cluster_id)) |>
    group_by(cluster_id) |>
    summarise(
      n_obs      = n(),
      n_days     = n_distinct(acq_dt),
      high_share = mean(conf_bucket == "high", na.rm = TRUE),
      mean_ndvi  = mean(ndvi, na.rm = TRUE),
      .groups = "drop"
    ) |>
    mutate(persistent = n_obs >= 5 & n_days >= 3)
  
  fires_veg <- fires_veg |>
    left_join(cluster_persistence, by = "cluster_id") |>
    mutate(
      cluster_type = case_when(
        !is.na(cluster_id) & persistent  ~ "persistent",
        !is.na(cluster_id) & !persistent ~ "transient",
        TRUE                             ~ "single"
      )
    )
} else {
  fires_veg <- fires_sf |> 
    filter(!likely_flare) |>
    mutate(cluster_id = NA_integer_,
           cluster_type = "single",
           n_obs = NA_integer_,
           n_days = NA_integer_,
           high_share = NA_real_,
           mean_ndvi = NA_real_,
           persistent = FALSE)
}

# Cluster industrial sources separately
fires_flare <- fires_sf |> filter(likely_flare)

if (nrow(fires_flare) > 0) {
  fires_flare_m <- st_transform(fires_flare, 3857)
  coords_flare_m <- st_coordinates(fires_flare_m)
  cl_flare <- dbscan(coords_flare_m, eps = 1000, minPts = 3)
  fires_flare$cluster_id <- ifelse(cl_flare$cluster == 0, NA_integer_, 
                                   cl_flare$cluster + 10000)  # Offset to avoid ID collision
  
  cluster_persistence_flare <- fires_flare |>
    st_drop_geometry() |>
    filter(!is.na(cluster_id)) |>
    group_by(cluster_id) |>
    summarise(
      n_obs      = n(),
      n_days     = n_distinct(acq_dt),
      high_share = mean(conf_bucket == "high", na.rm = TRUE),
      mean_ndvi  = mean(ndvi, na.rm = TRUE),
      .groups = "drop"
    ) |>
    mutate(persistent = n_obs >= 5 & n_days >= 3)
  
  fires_flare <- fires_flare |>
    left_join(cluster_persistence_flare, by = "cluster_id") |>
    mutate(
      cluster_type = case_when(
        !is.na(cluster_id) & persistent  ~ "persistent",
        !is.na(cluster_id) & !persistent ~ "transient",
        TRUE                             ~ "single"
      )
    )
} else {
  fires_flare <- fires_sf |> 
    filter(likely_flare) |>
    mutate(cluster_id = NA_integer_,
           cluster_type = "single",
           n_obs = NA_integer_,
           n_days = NA_integer_,
           high_share = NA_real_,
           mean_ndvi = NA_real_,
           persistent = FALSE)
}

# Combine both datasets
fires_sf <- bind_rows(fires_veg, fires_flare)

# Create combined classification (source type + cluster type)
fires_sf <- fires_sf |>
  mutate(
    fire_source = if_else(likely_flare, "industrial", "vegetation"),
    combined_type = paste0(fire_source, "_", cluster_type)
  )

cat("\n=== Fire Classification by Source and Cluster Type ===\n")
print(fires_sf |> 
        st_drop_geometry() |> 
        count(fire_source, cluster_type) |> 
        arrange(fire_source, cluster_type))

# -----------------------
# 9) Assign region & Arabic translation
# -----------------------
fires_sf <- st_join(fires_sf, saudi_states, join = st_within)
if (anyNA(fires_sf$region)) {
  miss_idx <- which(is.na(fires_sf$region))
  nearest  <- st_nearest_feature(fires_sf[miss_idx, ], saudi_states)
  fires_sf$region[miss_idx] <- saudi_states$region[nearest]
}

# Arabic region name mapping
region_names_arabic <- c(
  "Makkah" = "مكة المكرمة", "Makkah al Mukarramah" = "مكة المكرمة", "Makkah Province" = "مكة المكرمة",
  "Riyadh" = "الرياض", "Ar Riyad" = "الرياض", "Riyadh Province" = "الرياض",
  "Eastern Province" = "المنطقة الشرقية", "Ash Sharqiyah" = "المنطقة الشرقية", "Eastern" = "المنطقة الشرقية",
  "Asir" = "عسير", "`Asir" = "عسير", "Asir Province" = "عسير",
  "Medina" = "المدينة المنورة", "Al Madinah al Munawwarah" = "المدينة المنورة", 
  "Al Madinah" = "المدينة المنورة", "Madinah" = "المدينة المنورة", "Al Madinah Province" = "المدينة المنورة",
  "Al Madinah al Munawwarah Province" = "المدينة المنورة", "Madinah Province" = "المدينة المنورة",
  "Qasim" = "القصيم", "Al Qasim" = "القصيم", "Al Qassim" = "القصيم", "Al Quassim" = "القصيم",
  "Qassim" = "القصيم", "Quassim" = "القصيم", "Al Qassim Province" = "القصيم",
  "Qassim Province" = "القصيم", "Quassim Province" = "القصيم",
  "Al Qasim Province" = "القصيم", "Qasim Province" = "القصيم",
  "Tabuk" = "تبوك", "Tabuk Province" = "تبوك",
  "Jazan" = "جازان", "Jizan" = "جازان", "Jazan Province" = "جازان",
  "Ha'il" = "حائل", "Hail" = "حائل", "Ha'il Province" = "حائل",
  "Northern Borders" = "الحدود الشمالية", "Al Hudud ash Shamaliyah" = "الحدود الشمالية",
  "Northern Borders Province" = "الحدود الشمالية",
  "Najran" = "نجران", "Najran Province" = "نجران",
  "Al Bahah" = "الباحة", "Al Baha" = "الباحة", "Bahah" = "الباحة", "Al Bahah Province" = "الباحة",
  "Al Jawf" = "الجوف", "Al Jouf" = "الجوف", "Jawf" = "الجوف", "Al Jawf Province" = "الجوف"
)

translate_region <- function(region_name) {
  if (is.na(region_name)) return("غير محدد")
  region_str <- trimws(as.character(region_name))
  # Try exact match first
  translated <- region_names_arabic[region_str]
  if (!is.na(translated)) return(as.character(translated))
  # Try case-insensitive match
  region_lower <- tolower(region_str)
  for (i in seq_along(region_names_arabic)) {
    if (tolower(trimws(names(region_names_arabic)[i])) == region_lower) {
      return(as.character(region_names_arabic[i]))
    }
  }
  # Try partial match (contains)
  for (i in seq_along(region_names_arabic)) {
    key_lower <- tolower(trimws(names(region_names_arabic)[i]))
    if (grepl(key_lower, region_lower, fixed = TRUE) || grepl(region_lower, key_lower, fixed = TRUE)) {
      return(as.character(region_names_arabic[i]))
    }
  }
  # If still not found, return original
  return(region_str)
}

# Add Arabic region names
fires_sf <- fires_sf |>
  mutate(region_ar = sapply(region, translate_region))

saudi_states <- saudi_states |>
  mutate(region_ar = sapply(region, translate_region))

# -----------------------
# 10) Region centers
# -----------------------
state_centers <- suppressWarnings(st_point_on_surface(saudi_states)) |>
  mutate(lat = st_coordinates(geometry)[,2],
         lon = st_coordinates(geometry)[,1]) |>
  select(region, region_ar, lat, lon)

# -----------------------
# 11) Weather Data for FWI (Enhanced)
# -----------------------
cat("\nFetching weather data for FWI calculation...\n")

get_weather_data <- function(lat, lon, days_back = 7) {
  date_end   <- Sys.Date()
  date_start <- date_end - days_back
  
  # Get comprehensive weather data
  url <- paste0(
    "https://api.open-meteo.com/v1/forecast?",
    "latitude=", lat, "&longitude=", lon,
    "&hourly=temperature_2m,relative_humidity_2m,precipitation,windspeed_10m",
    "&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,",
    "windspeed_10m_max,et0_fao_evapotranspiration",
    "&timezone=UTC&past_days=", days_back
  )
  
  resp <- request(url) |> req_perform() |> resp_body_json(simplifyVector = TRUE)
  
  if (is.null(resp$daily)) {
    return(tibble(
      temp_max = NA_real_, temp_min = NA_real_, 
      precip = NA_real_, wind_max = NA_real_,
      rh_min = NA_real_, evap = NA_real_
    ))
  }
  
  # Daily data for FWI
  daily <- tibble(
    date = as.Date(resp$daily$time),
    temp_max = as.numeric(resp$daily$temperature_2m_max),
    temp_min = as.numeric(resp$daily$temperature_2m_min),
    precip = as.numeric(resp$daily$precipitation_sum),
    wind_max = as.numeric(resp$daily$windspeed_10m_max),
    evap = as.numeric(resp$daily$et0_fao_evapotranspiration)
  )
  
  # Get minimum relative humidity from hourly data
  if (!is.null(resp$hourly)) {
    hourly_rh <- tibble(
      time = ymd_hm(resp$hourly$time, tz = "UTC"),
      rh = as.numeric(resp$hourly$relative_humidity_2m)
    ) |>
      mutate(date = as.Date(time)) |>
      group_by(date) |>
      summarise(rh_min = min(rh, na.rm = TRUE), .groups = "drop")
    
    daily <- daily |> left_join(hourly_rh, by = "date")
  } else {
    daily$rh_min <- NA_real_
  }
  
  # Get most recent values
  daily |>
    arrange(desc(date)) |>
    slice(1)
}

# Fetch weather for all regions
weather_list <- lapply(seq_len(nrow(state_centers)), function(i) {
  nm <- state_centers$region[i]
  nm_ar <- state_centers$region_ar[i]
  la <- state_centers$lat[i]
  lo <- state_centers$lon[i]
  
  cat(sprintf("Fetching weather for %s (%s)...\n", nm_ar, nm))
  
  weather <- get_weather_data(la, lo, days_back = 7)
  weather$region <- nm
  weather$region_ar <- nm_ar
  
  Sys.sleep(0.5)  # Rate limiting
  weather
})

weather_df <- bind_rows(weather_list)

# -----------------------
# 12) Fire Weather Index (FWI) - Adapted for Saudi Arabia
# -----------------------
cat("\nCalculating Fire Weather Index (FWI)...\n")

# Saudi Arabia adaptations:
# 1. Arid climate adjustment - lower moisture thresholds
# 2. High temperature normalization (40°C+ is common)
# 3. Low humidity emphasis (critical in desert)
# 4. Wind factor (shamal winds)
# 5. Seasonal vegetation cycles (sparse but important)

calculate_fwi_saudi <- function(temp_max, rh_min, wind_max, precip, 
                                precip_7d = 0, ndvi = 0.1) {
  
  # Handle missing values with Saudi defaults
  temp_max <- ifelse(is.na(temp_max) | is.infinite(temp_max), 35, temp_max)
  rh_min   <- ifelse(is.na(rh_min) | is.infinite(rh_min), 20, rh_min)
  wind_max <- ifelse(is.na(wind_max) | is.infinite(wind_max), 15, wind_max)
  precip   <- ifelse(is.na(precip) | is.infinite(precip), 0, precip)
  precip_7d <- ifelse(is.na(precip_7d) | is.infinite(precip_7d), 0, precip_7d)
  ndvi     <- ifelse(is.na(ndvi) | is.infinite(ndvi), 0.1, ndvi)
  
  # 1. TEMPERATURE COMPONENT (Saudi-adapted)
  # Extreme heat common in Saudi Arabia - normalize differently
  # Risk increases dramatically above 35°C
  temp_score <- case_when(
    temp_max < 25  ~ 0,
    temp_max < 35  ~ (temp_max - 25) / 10 * 30,      # 0-30 points
    temp_max < 45  ~ 30 + (temp_max - 35) / 10 * 40, # 30-70 points
    TRUE           ~ 70 + min((temp_max - 45) / 5 * 30, 30) # 70-100 points
  )
  
  # 2. HUMIDITY COMPONENT (Critical in arid climates)
  # Saudi Arabia: RH often <20% - this is VERY dangerous for fires
  rh_score <- case_when(
    rh_min > 50   ~ 0,
    rh_min > 30   ~ (50 - rh_min) / 20 * 30,      # 0-30 points
    rh_min > 15   ~ 30 + (30 - rh_min) / 15 * 40, # 30-70 points
    TRUE          ~ 70 + min((15 - rh_min) / 10 * 30, 30) # 70-100 points
  )
  
  # 3. WIND COMPONENT (Shamal winds)
  # Strong winds common in Saudi Arabia, especially winter/spring
  wind_score <- case_when(
    wind_max < 10  ~ wind_max / 10 * 20,           # 0-20 points
    wind_max < 30  ~ 20 + (wind_max - 10) / 20 * 50, # 20-70 points
    TRUE           ~ 70 + min((wind_max - 30) / 20 * 30, 30) # 70-100 points
  )
  
  # 4. DROUGHT/PRECIPITATION COMPONENT (Water scarcity)
  # In Saudi Arabia, any rain is significant
  # Long dry periods are normal, so emphasize recent rain impact
  days_since_rain <- ifelse(precip > 0.1, 0, 
                            ifelse(precip_7d > 0.1, 3, 7))
  
  drought_score <- case_when(
    days_since_rain == 0     ~ 0,          # Recent rain
    days_since_rain <= 3     ~ 20,         # Rain in last 3 days
    days_since_rain <= 7     ~ 50,         # Rain in last week
    precip_7d > 5            ~ 40,         # Moderate recent rain
    precip_7d > 1            ~ 60,         # Light recent rain
    TRUE                     ~ 80          # Prolonged drought (normal)
  )
  
  # 5. FUEL/VEGETATION COMPONENT (NDVI-based)
  # Even sparse vegetation can burn in arid climates
  # Lower NDVI threshold for Saudi Arabia
  fuel_score <- case_when(
    ndvi < 0.1   ~ 5,          # Minimal fuel (barren)
    ndvi < 0.15  ~ 15,         # Sparse fuel (can still burn)
    ndvi < 0.25  ~ 40,         # Moderate fuel (significant risk)
    ndvi < 0.4   ~ 70,         # Good fuel load
    TRUE         ~ 90          # Dense vegetation (rare but extreme risk)
  )
  
  # 6. COMPOSITE FWI (Weighted for Saudi conditions)
  # Higher weight on humidity and temperature (arid climate)
  fwi_raw <- (
    temp_score * 0.25 +      # Temperature (critical in extreme heat)
      rh_score * 0.30 +        # Humidity (MOST critical in desert)
      wind_score * 0.20 +      # Wind (shamal effect)
      drought_score * 0.15 +   # Drought (always present, less weight)
      fuel_score * 0.10        # Fuel availability (sparse vegetation)
  )
  
  # Normalize to 0-100 scale
  fwi <- pmin(pmax(fwi_raw, 0), 100)
  
  return(fwi)
}

# Calculate FWI for each region
weather_df <- weather_df |>
  mutate(
    fwi = calculate_fwi_saudi(
      temp_max = temp_max,
      rh_min = rh_min,
      wind_max = wind_max,
      precip = precip,
      precip_7d = precip,  # Simplified - could fetch 7-day sum
      ndvi = 0.15  # Regional average, will be refined with actual data
    ),
    fire_danger = case_when(
      fwi < 20  ~ "منخفض (Low)",
      fwi < 40  ~ "متوسط (Moderate)",
      fwi < 60  ~ "عالي (High)",
      fwi < 80  ~ "عالي جداً (Very High)",
      TRUE      ~ "شديد الخطورة (Extreme)"
    )
  )

cat("\n=== Fire Weather Index by Region ===\n")
print(weather_df |>
        select(region, fwi, fire_danger, temp_max, rh_min, wind_max) |>
        arrange(desc(fwi)))

# -----------------------
# 13) Enhanced KRI with FWI
# -----------------------
zscore <- function(x) {
  if (all(is.na(x))) return(rep(NA_real_, length(x)))
  as.numeric(scale(x))
}

kri <- fires_sf |>
  mutate(
    is_high      = conf_bucket == "high",
    is_transient = cluster_type == "transient",
    is_veg_fire  = !likely_flare & ndvi >= 0.1  # Lower threshold for Saudi
  ) |>
  st_drop_geometry() |>
  group_by(region, region_ar) |>
  summarise(
    fires_7d              = n(),
    fires_high_7d         = sum(is_high, na.rm = TRUE),
    veg_fires_7d          = sum(is_veg_fire, na.rm = TRUE),
    flares_7d             = sum(likely_flare, na.rm = TRUE),
    clusters_transient_7d = n_distinct(cluster_id[is_transient], na.rm = TRUE),
    mean_frp_high         = mean(frp[is_high], na.rm = TRUE),
    mean_ndvi             = mean(ndvi[!likely_flare], na.rm = TRUE),
    .groups = "drop"
  ) |>
  left_join(weather_df |> select(region, fwi, fire_danger, temp_max, 
                                 rh_min, wind_max), by = "region") |>
  mutate(
    # Ensure region_ar is set (should already be from grouping, but add fallback)
    region_ar = coalesce(region_ar, sapply(region, translate_region)),
    # Enhanced KRI with FWI as primary component
    risk_index = 
      0.40 * zscore(fwi) +                    # FWI is primary predictor
      0.25 * zscore(veg_fires_7d) +           # Actual fire activity
      0.20 * zscore(fires_high_7d) +          # High-confidence fires
      0.10 * zscore(clusters_transient_7d) +  # Growing fires
      0.05 * zscore(mean_ndvi)                # Fuel availability
  ) |>
  arrange(desc(risk_index))

saudi_states_kri <- saudi_states |>
  left_join(kri |> select(-region_ar), by = "region") |>
  mutate(
    # Ensure region_ar is set (from saudi_states, with fallback translation)
    region_ar = coalesce(region_ar, sapply(region, translate_region))
  )

# Attach FWI to fire points
fires_sf <- fires_sf |>
  left_join(weather_df |> select(region, region_ar, fwi, fire_danger, temp_max, 
                                 rh_min, wind_max), by = "region")

# -----------------------
# 14) Diagnostics
# -----------------------
daily_trend <- fires_sf |>
  st_drop_geometry() |>
  count(acq_dt, ndvi_class) |>
  arrange(acq_dt)

cat("\n=== Enhanced KRI Summary (with FWI) ===\n")
print(kri)

# -----------------------
# 15) Palettes & helpers
# -----------------------
# New color scheme: 
# - Vegetation fires: Red/Orange/Yellow tones (warm colors for actual fires)
# - Industrial sources: Purple/Pink tones (distinct from natural fires)
# - Intensity: Darker = more persistent, Lighter = single detection

type_pal <- colorFactor(
  palette = c(
    # Vegetation fires (red/orange tones)
    "vegetation_persistent" = "#8B0000",  # Dark red - persistent wildfire
    "vegetation_transient"  = "#E24A33",  # Orange-red - spreading fire
    "vegetation_single"     = "#FFA500",  # Orange - single detection
    
    # Industrial sources (purple/magenta tones)
    "industrial_persistent" = "#6A0DAD",  # Dark purple - persistent flare
    "industrial_transient"  = "#9B59B6",  # Medium purple - transient industrial
    "industrial_single"     = "#E91E63"   # Pink - single industrial detection
  ),
  domain = c("vegetation_persistent", "vegetation_transient", "vegetation_single",
             "industrial_persistent", "industrial_transient", "industrial_single")
)

risk_pal <- colorBin(
  "YlOrRd",
  domain = saudi_states_kri$risk_index,
  bins   = c(-Inf,-1,-0.3,0.3,1,Inf),
  na.color = "#E0E0E0"
)

fwi_pal <- colorBin(
  palette = c("#00FF00","#FFFF00","#FFA500","#FF0000","#8B0000"),
  domain = c(0, 100),
  bins = c(0, 20, 40, 60, 80, 100)
)

popup_html <- function(rec) {
  f <- names(rec)
  g <- function(k) if (k %in% f) as.character(rec[[k]]) else NA_character_
  lat <- as.numeric(g("latitude")); lon <- as.numeric(g("longitude"))
  ndvi_val <- as.numeric(g("ndvi"))
  ndvi_display <- if (!is.na(ndvi_val)) sprintf("%.3f", ndvi_val) else "—"
  fwi_val <- as.numeric(g("fwi"))
  fwi_display <- if (!is.na(fwi_val)) sprintf("%.1f", fwi_val) else "—"
  
  # ML classification info
  flare_prob <- as.numeric(g("flare_probability"))
  flare_prob_display <- if (!is.na(flare_prob)) sprintf("%.1f%%", flare_prob * 100) else "—"
  is_flare <- g("likely_flare") == "TRUE"
  classification_color <- if (is_flare) "#f3e5f5" else "#ffebee"  # Purple tint for industrial, red for veg
  classification_icon <- if (is_flare) "🏭" else "🔥"
  classification_text <- if (is_flare) "مصدر صناعي" else "حريق نباتي"
  
  # Get cluster type and combined classification
  cluster_type <- g("cluster_type")
  fire_source <- g("fire_source")
  
  # Create display text for cluster type
  cluster_display <- switch(cluster_type,
    "persistent" = "مستمر (Persistent)",
    "transient" = "عابر (Transient)",
    "single" = "كشف واحد (Single)",
    cluster_type
  )
  
  sprintf(
    "<div style='font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial'>
      <div><strong>📅 التاريخ:</strong> %s %s UTC</div>
      <div><strong>📡 المستشعر:</strong> %s | <strong>القمر:</strong> %s</div>
      <div><strong>🔥 الثقة:</strong> %s | <strong>FRP:</strong> %s MW</div>
      <div><strong>المنطقة:</strong> %s</div>
      <div style='background-color:%s;padding:4px;margin:4px 0;border-radius:4px'>
        <strong>%s المصدر:</strong> %s (احتمال: %s)
      </div>
      <div style='background-color:#e3f2fd;padding:4px;margin:4px 0;border-radius:4px'>
        <strong>📊 نوع التجمع:</strong> %s
      </div>
      <div style='background-color:#e8f5e9;padding:4px;margin:4px 0;border-radius:4px'>
        <strong>🌿 NDVI:</strong> %s | <strong>التصنيف:</strong> %s
      </div>
      <div style='background-color:#fff3cd;padding:4px;margin:4px 0;border-radius:4px'>
        <strong>🌡️ FWI:</strong> %s | <strong>خطر الحريق:</strong> %s
      </div>
      <div><strong>🌡️ درجة الحرارة:</strong> %s°C | <strong>💨 رياح:</strong> %s م/ث</div>
      <div><strong>💧 رطوبة:</strong> %s%%</div>
      <div><strong>📍 الإحداثيات:</strong> %.4f, %.4f</div>
    </div>",
    g("acq_dt"), g("acq_time_fmt"),
    g("sensor"), g("satellite"),
    g("conf_label"), g("frp"),
    ifelse("region_ar" %in% names(rec), g("region_ar"), g("region")),
    classification_color, classification_icon, classification_text, flare_prob_display,
    cluster_display,
    ndvi_display, g("ndvi_class"),
    fwi_display, g("fire_danger"),
    ifelse(is.na(g("temp_max")), "—", sprintf("%.1f", as.numeric(g("temp_max")))),
    ifelse(is.na(g("wind_max")), "—", sprintf("%.1f", as.numeric(g("wind_max")))),
    ifelse(is.na(g("rh_min")), "—", sprintf("%.0f", as.numeric(g("rh_min")))),
    lat, lon
  )
}

make_labels <- function(dat) {
  if (nrow(dat) == 0) return(NULL)
  
  # Extract data as data frame to avoid geometry issues
  dat_df <- dat
  if ("sf" %in% class(dat)) {
    dat_df <- st_drop_geometry(dat)
  }
  
  lapply(seq_len(nrow(dat_df)), function(i) {
    r <- dat_df[i, , drop = FALSE]
    
    # Safe extraction with defaults
    region <- if ("region_ar" %in% names(r)) {
      as.character(r$region_ar[1])
    } else if ("region" %in% names(r)) {
      translate_region(as.character(r$region[1]))
    } else {
      "غير محدد"
    }
    cluster_type <- if ("cluster_type" %in% names(r)) as.character(r$cluster_type[1]) else "single"
    fire_source <- if ("fire_source" %in% names(r)) as.character(r$fire_source[1]) else "unknown"
    
    acq_dt <- if ("acq_dt" %in% names(r)) as.character(r$acq_dt[1]) else ""
    acq_time_fmt <- if ("acq_time_fmt" %in% names(r)) {
      time_val <- r$acq_time_fmt[1]
      if (is.na(time_val)) "00:00" else as.character(time_val)
    } else "00:00"
    
    ndvi_val <- if ("ndvi" %in% names(r)) r$ndvi[1] else NA_real_
    ndvi_display <- if (!is.na(ndvi_val)) sprintf("%.3f", ndvi_val) else "—"
    
    fwi_val <- if ("fwi" %in% names(r)) r$fwi[1] else NA_real_
    fwi_display <- if (!is.na(fwi_val)) sprintf("%.0f", fwi_val) else "—"
    
    fire_danger <- if ("fire_danger" %in% names(r)) as.character(r$fire_danger[1]) else "—"
    
    frp_val <- if ("frp" %in% names(r)) r$frp[1] else NA_real_
    frp_display <- if (!is.na(frp_val)) sprintf("%.1f MW", frp_val) else "—"
    
    conf_val <- if ("conf_label" %in% names(r)) as.character(r$conf_label[1]) else "—"
    
    # Source icon and text
    source_icon <- if (fire_source == "industrial") "🏭" else "🔥"
    source_text <- if (fire_source == "industrial") "Industrial" else "Vegetation"
    
    # Cluster type display
    cluster_display <- switch(cluster_type,
      "persistent" = "Persistent",
      "transient" = "Transient",
      "single" = "Single",
      cluster_type
    )
    
    ml_prob <- if ("flare_probability" %in% names(r)) {
      sprintf("%.0f%%", r$flare_probability[1] * 100)
    } else "—"
    
    HTML(sprintf(
      "<div style='font-family:system-ui;-webkit-font-smoothing:antialiased;min-width:200px;'>
         <div style='font-weight:bold;font-size:14px;margin-bottom:5px;'>%s</div>
         <div style='background:#f8f9fa;padding:5px;border-radius:3px;margin:3px 0;'>
           <div>📅 %s %s UTC</div>
           <div>🔥 FRP: %s | Conf: %s</div>
           <div>🌿 NDVI: %s</div>
         </div>
         <div style='background:#e3f2fd;padding:5px;border-radius:3px;margin:3px 0;'>
           <div>%s %s | %s</div>
         </div>
         <div style='background:#fff3cd;padding:5px;border-radius:3px;margin:3px 0;'>
           <div>🌡️ FWI: %s | %s</div>
         </div>
       </div>",
      region, acq_dt, acq_time_fmt, frp_display, conf_val,
      ndvi_display,
      source_icon, source_text, cluster_display,
      fwi_display, fire_danger
    ))
  })
}

kri_head <- kri |>
  mutate(
    veg_fires_7d = ifelse(is.na(veg_fires_7d), 0, veg_fires_7d),
    fwi          = round(fwi, 1),
    risk_index   = round(risk_index, 2),
    region_display = ifelse(!is.na(region_ar), region_ar, region)
  ) |>
  slice_head(n = 5)

kri_table_html <- paste0(
  "<div style='font-family:system-ui; font-size:12px'>",
  "<div style='margin-bottom:6px'><strong>أعلى 5 مناطق - مؤشر المخاطر (KRI + FWI)</strong></div>",
  "<table style='border-collapse:collapse'>",
  "<tr><th style='text-align:right;padding:2px 6px'>المنطقة</th>",
  "<th style='text-align:center;padding:2px 6px'>KRI</th>",
  "<th style='text-align:center;padding:2px 6px'>FWI</th>",
  "<th style='text-align:center;padding:2px 6px'>🔥 نباتية</th>",
  "<th style='text-align:center;padding:2px 6px'>خطر الحريق</th></tr>",
  paste(
    apply(kri_head, 1, function(row) {
      sprintf("<tr>
        <td style='padding:2px 6px'>%s</td>
        <td style='text-align:center;padding:2px 6px;font-weight:bold'>%s</td>
        <td style='text-align:center;padding:2px 6px'>%s</td>
        <td style='text-align:center;padding:2px 6px'>%s</td>
        <td style='text-align:center;padding:2px 6px;font-size:10px'>%s</td>
      </tr>",
              row[["region_display"]], row[["risk_index"]], row[["fwi"]],
              row[["veg_fires_7d"]], row[["fire_danger"]])
    }),
    collapse = ""
  ),
  "</table></div>"
)

# -----------------------
# 16) Feedback Analysis Dashboard
# -----------------------

analyze_feedback <- function() {
  if (!file.exists(FEEDBACK_FILE)) {
    cat("\n=== No Feedback Data Available ===\n")
    cat("Start adding feedback using add_feedback() function\n")
    return(invisible(NULL))
  }
  
  feedback <- read_csv(FEEDBACK_FILE, show_col_types = FALSE)
  
  if (nrow(feedback) == 0) {
    cat("\n=== No Feedback Entries Yet ===\n")
    cat("Start adding feedback using add_feedback() function\n")
    return(invisible(NULL))
  }
  
  cat("\n=== Feedback Analysis Dashboard ===\n")
  cat(sprintf("Total feedback entries: %d\n", nrow(feedback)))
  cat(sprintf("Date range: %s to %s\n", 
              min(feedback$feedback_date), max(feedback$feedback_date)))
  
  # Agreement analysis
  feedback <- feedback |>
    mutate(
      ml_agreed = (ml_prediction == user_label),
      confidence_bucket = cut(ml_probability, 
                              breaks = c(0, 0.6, 0.8, 1.0),
                              labels = c("low", "medium", "high"),
                              include.lowest = TRUE)
    )
  
  cat(sprintf("\n✓ ML Agreement Rate: %.1f%%\n", 
              100 * mean(feedback$ml_agreed)))
  cat(sprintf("  Correct predictions: %d\n", sum(feedback$ml_agreed)))
  cat(sprintf("  Incorrect predictions: %d\n", sum(!feedback$ml_agreed)))
  
  # Confusion matrix
  cat("\n--- Confusion Matrix ---\n")
  confusion <- feedback |>
    count(ml_prediction, user_label)
  print(confusion)
  
  # Errors by confidence
  if (sum(!feedback$ml_agreed) > 0) {
    cat("\n--- Errors by Confidence Level ---\n")
    error_analysis <- feedback |>
      filter(!ml_agreed) |>
      count(confidence_bucket, ml_prediction, user_label)
    print(error_analysis)
    
    # Most common mistakes
    cat("\n--- Common Error Patterns (Top 10) ---\n")
    errors <- feedback |>
      filter(!ml_agreed) |>
      select(latitude, longitude, ml_prediction, ml_probability, user_label, notes)
    
    print(head(errors, 10))
    
    # Feature analysis for errors
    if (exists("fires_with_features")) {
      cat("\n--- Feature Analysis of Errors ---\n")
      error_coords <- errors |>
        mutate(
          lat_rounded = round(latitude, 4),
          lon_rounded = round(longitude, 4)
        )
      
      error_features <- fires_with_features |>
        mutate(
          lat_rounded = round(latitude, 4),
          lon_rounded = round(longitude, 4)
        ) |>
        semi_join(error_coords, by = c("lat_rounded", "lon_rounded")) |>
        select(latitude, longitude, mean_ndvi, spatial_stability, 
               detection_rate, frp_stability, n_detections)
      
      if (nrow(error_features) > 0) {
        cat("\nError cases feature summary:\n")
        print(summary(error_features |> select(-latitude, -longitude)))
      }
    }
  } else {
    cat("\n✓ Perfect agreement - no errors!\n")
  }
  
  # User label distribution
  cat("\n--- User Label Distribution ---\n")
  label_dist <- feedback |>
    count(user_label) |>
    mutate(percentage = n / sum(n) * 100)
  print(label_dist)
  
  return(invisible(feedback))
}

# Run feedback analysis if data exists
if (file.exists(FEEDBACK_FILE)) {
  feedback_analysis <- analyze_feedback()
}

# -----------------------
# 17) Time slices
# -----------------------
fires_sf <- fires_sf |>
  mutate(
    hhmm = if_else(!is.na(acq_time_utc),
                   paste0(substr(acq_time_utc,1,2), ":", substr(acq_time_utc,3,4)),
                   "00:00"),
    timestamp = as.POSIXct(paste(acq_dt, hhmm), tz = "UTC")
  )

now_utc  <- as.POSIXct(Sys.time(), tz = "UTC")
last_24h <- fires_sf |> filter(timestamp >= now_utc - 24*3600)
last_7d  <- fires_sf

# Separate vegetation fires and flares for optional layers
veg_fires_24h <- last_24h |> filter(!likely_flare)
flares_24h    <- last_24h |> filter(likely_flare)
veg_fires_7d  <- last_7d |> filter(!likely_flare)
flares_7d     <- last_7d |> filter(likely_flare)

days       <- sort(unique(fires_sf$acq_dt))
day_groups <- paste0("اليوم: ", days)

cat(sprintf("\n=== Data Summary for Map ===\n"))
cat(sprintf("Total detections: %d\n", nrow(fires_sf)))
cat(sprintf("Last 24h: %d (Veg: %d, Flares: %d)\n", 
            nrow(last_24h), nrow(veg_fires_24h), nrow(flares_24h)))
cat(sprintf("Last 7d: %d (Veg: %d, Flares: %d)\n", 
            nrow(last_7d), nrow(veg_fires_7d), nrow(flares_7d)))
cat(sprintf("Unique days: %d\n", length(days)))

# -----------------------
# 18) Map
# -----------------------
center <- st_point(c(mean(c(bb["xmin"], bb["xmax"])),
                     mean(c(bb["ymin"], bb["ymax"])))) |>
  st_sfc(crs = 4326) |> st_coordinates()
last_update <- paste0("آخر تحديث: ", format(Sys.time(), "%Y-%m-%d %H:%M UTC"))

m <- leaflet(options = leafletOptions(minZoom = 4)) |>
  addProviderTiles(providers$CartoDB.Positron,  group = "خريطة مبسطة") |>
  addProviderTiles(providers$Esri.WorldImagery, group = "صور الأقمار") |>
  addProviderTiles(providers$Esri.WorldTopoMap, group = "طبوغرافية") |>
  addTiles(urlTemplate = "https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}", 
           group = "Google Maps",
           attribution = 'Map data ©2024 Google') |>
  addTiles(urlTemplate = "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}", 
           group = "Google Satellite",
           attribution = 'Imagery ©2024 Google') |>
  addTiles(urlTemplate = "https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}", 
           group = "Google Hybrid",
           attribution = 'Map data ©2024 Google') |>
  addPolygons(data = saudi, weight = 2, color = "#2D3E50", fill = FALSE,
              opacity = 1, group = "حدود المملكة") |>
  addPolygons(data = saudi_states, weight = 1, color = "#777", fill = FALSE,
              opacity = 0.7, group = "المناطق")

# KRI choropleth
m <- m |>
  addPolygons(
    data = saudi_states_kri,
    fillColor = ~risk_pal(risk_index), fillOpacity = 0.55,
    color = "#444", weight = 0.7,
    group = "مؤشر المخاطر الشامل (KRI)",
    label = ~HTML(sprintf(
      "<div style='font-family:system-ui'>%s<br/>KRI: %s | FWI: %s<br/>%s</div>",
      ifelse(!is.na(region_ar), region_ar, region), 
      ifelse(is.na(risk_index), "—", round(risk_index,2)),
      ifelse(is.na(fwi), "—", round(fwi,1)),
      fire_danger
    )),
    highlight = highlightOptions(weight = 1.5, color = "#333", bringToFront = TRUE)
  ) |>
  addLegend("bottomleft", pal = risk_pal, values = saudi_states_kri$risk_index,
            title = "مؤشر المخاطر الشامل<br/>(منخفض ← مرتفع)", 
            opacity = 0.8,
            labFormat = labelFormat(suffix = ""),
            group = "مؤشر المخاطر الشامل (KRI)")

# FWI choropleth
m <- m |>
  addPolygons(
    data = saudi_states_kri,
    fillColor = ~fwi_pal(fwi), fillOpacity = 0.65,
    color = "#444", weight = 0.7,
    group = "مؤشر طقس الحريق (FWI)",
    label = ~HTML(sprintf(
      "<div style='font-family:system-ui'>%s<br/>FWI: %s<br/>%s</div>",
      ifelse(!is.na(region_ar), region_ar, region), 
      ifelse(is.na(fwi), "—", round(fwi,1)),
      fire_danger
    )),
    highlight = highlightOptions(weight = 1.5, color = "#333", bringToFront = TRUE)
  ) |>
  addLegend("bottomleft", pal = fwi_pal, values = c(0, 100),
            title = "مؤشر طقس الحريق<br/>(منخفض ← مرتفع)", 
            opacity = 0.8,
            labFormat = labelFormat(suffix = ""),
            group = "مؤشر طقس الحريق (FWI)")

add_time_layer <- function(map_obj, dat, group_name) {
  if (nrow(dat) == 0) {
    cat(sprintf("Warning: Layer '%s' has 0 data points\n", group_name))
    return(map_obj)
  }
  
  map_obj |>
    addCircleMarkers(
      data = dat, group = group_name,
      lng = ~longitude, lat = ~latitude,
      radius = 5, stroke = TRUE, weight = 1, color = "#222",
      fillColor = ~type_pal(combined_type), fillOpacity = 0.9,
      popup = lapply(split(dat, seq_len(nrow(dat))), popup_html),
      label = make_labels(dat),
      labelOptions = labelOptions(noHide = FALSE, direction = "auto", opacity = 0.95)
    )
}

# Add ALL fires layers (including flares) as primary layers
m <- m |>
  add_time_layer(last_24h, "جميع الكشوفات - آخر 24 ساعة") |>
  add_time_layer(last_7d,  "جميع الكشوفات - آخر 7 أيام")

# Add vegetation-only layers as optional
m <- m |>
  add_time_layer(veg_fires_24h, "حرائق نباتية فقط - آخر 24 ساعة") |>
  add_time_layer(veg_fires_7d,  "حرائق نباتية فقط - آخر 7 أيام")

# Add flare-only layers as optional
m <- m |>
  add_time_layer(flares_24h, "توهجات صناعية فقط - آخر 24 ساعة") |>
  add_time_layer(flares_7d,  "توهجات صناعية فقط - آخر 7 أيام")

for (i in seq_along(days)) {
  d  <- days[i]
  gp <- day_groups[i]
  subd <- fires_sf |> filter(acq_dt == d)  # Show ALL fires, not just vegetation
  
  # Skip if no data for this day
  if (nrow(subd) > 0) {
    m <- m |>
      addCircleMarkers(
        data = subd, group = gp,
        lng = ~longitude, lat = ~latitude,
        radius = 5, stroke = TRUE, weight = 1, color = "#222",
        fillColor = ~type_pal(combined_type), fillOpacity = 0.9,
        popup = lapply(split(subd, seq_len(nrow(subd))), popup_html),
        label = make_labels(subd),
        labelOptions = labelOptions(noHide = FALSE, direction = "auto", opacity = 0.95)
      )
  }
}

# Create custom HTML legend for fire classification (fully in Arabic)
fire_legend_html <- tags$div(
  style = "background:white;padding:10px;border-radius:5px;box-shadow:0 2px 8px rgba(0,0,0,.3);font-size:12px;min-width:220px;z-index:1000;",
  tags$div(
    style = "font-weight:bold;margin-bottom:8px;text-align:center;font-size:14px;color:#333;",
    "تصنيف الحرائق"
  ),
  tags$div(
    style = "display:flex;align-items:center;margin:4px 0;",
    tags$span(style = "display:inline-block;width:20px;height:20px;background:#8B0000;border:2px solid #222;margin-left:5px;border-radius:3px;"),
    tags$span("🔥 حريق نباتي - مستمر")
  ),
  tags$div(
    style = "display:flex;align-items:center;margin:4px 0;",
    tags$span(style = "display:inline-block;width:20px;height:20px;background:#E24A33;border:2px solid #222;margin-left:5px;border-radius:3px;"),
    tags$span("🔥 حريق نباتي - عابر")
  ),
  tags$div(
    style = "display:flex;align-items:center;margin:4px 0;",
    tags$span(style = "display:inline-block;width:20px;height:20px;background:#FFA500;border:2px solid #222;margin-left:5px;border-radius:3px;"),
    tags$span("🔥 حريق نباتي - كشف واحد")
  ),
  tags$div(
    style = "display:flex;align-items:center;margin:4px 0;",
    tags$span(style = "display:inline-block;width:20px;height:20px;background:#6A0DAD;border:2px solid #222;margin-left:5px;border-radius:3px;"),
    tags$span("🏭 مصدر صناعي - مستمر")
  ),
  tags$div(
    style = "display:flex;align-items:center;margin:4px 0;",
    tags$span(style = "display:inline-block;width:20px;height:20px;background:#9B59B6;border:2px solid #222;margin-left:5px;border-radius:3px;"),
    tags$span("🏭 مصدر صناعي - عابر")
  ),
  tags$div(
    style = "display:flex;align-items:center;margin:4px 0;",
    tags$span(style = "display:inline-block;width:20px;height:20px;background:#E91E63;border:2px solid #222;margin-left:5px;border-radius:3px;"),
    tags$span("🏭 مصدر صناعي - كشف واحد")
  )
)

m |>
  addControl(
    html = fire_legend_html,
    position = "bottomleft"
  ) |>
  addControl(
    html = tags$div(
      style="background:white;padding:8px 10px;border-radius:8px;box-shadow:0 1px 4px rgba(0,0,0,.25);max-width:450px",
      HTML(kri_table_html)
    ),
    position = "topright"
  ) |>
  addControl(
    html = tags$div(
      style="background:white;padding:6px 10px;border-radius:8px;box-shadow:0 1px 4px rgba(0,0,0,.25);",
      HTML(sprintf("<div><strong>%s</strong></div>
                    <div style='font-size:11px;margin-top:4px'>
                    نظام إنذار مبكر للحرائق | ML + FWI + NDVI
                    </div>
                    <div style='font-size:10px;margin-top:2px;color:#666'>
                    🤖 تصنيف ذكي للتوهجات الصناعية
                    </div>
                    <div style='font-size:10px;margin-top:4px;color:#007bff;cursor:pointer;' 
                         onclick='alert(\"Annotation Mode: Click any fire point, then use browser console to run:\\n\\nadd_annotation(lat, lon, \\\"flare\\\") or\\nadd_annotation(lat, lon, \\\"wildfire\\\")\")'>
                    📝 تعليمات التصنيف
                    </div>", last_update))
    ),
    position = "topleft"
  ) |>
  htmlwidgets::onRender("
    function(el, x) {
      var annotationMode = false;
      var annotations = [];
      
      // Create annotation button
      var button = L.control({position: 'topleft'});
      button.onAdd = function(map) {
        var div = L.DomUtil.create('div', 'annotation-control');
        div.innerHTML = '<div style=\"background:white;padding:15px;border-radius:8px;box-shadow:0 2px 10px rgba(0,0,0,0.3);\">' +
          '<button id=\"annotateBtn\" style=\"background:#28a745;color:white;padding:12px 24px;border:none;border-radius:5px;cursor:pointer;font-weight:bold;width:100%;margin-bottom:10px;font-size:14px;\">🎯 تفعيل وضع التصنيف</button>' +
          '<div id=\"counterDiv\" style=\"display:none;background:#f8f9fa;padding:10px;border-radius:5px;margin-bottom:10px;text-align:center;\">' +
          '<div style=\"font-size:24px;font-weight:bold;color:#28a745;\" id=\"counterNum\">0</div>' +
          '<div style=\"font-size:12px;color:#666;\">تصنيفات</div>' +
          '</div>' +
          '<button id=\"downloadBtn\" style=\"background:#007bff;color:white;padding:10px 20px;border:none;border-radius:5px;cursor:pointer;font-weight:bold;width:100%;display:none;font-size:14px;\">💾 تحميل جميع التصنيفات</button>' +
          '</div>';
        div.style.background = 'transparent';
        div.style.border = 'none';
        return div;
      };
      button.addTo(this);
      
      // Toggle annotation mode
      document.getElementById('annotateBtn').onclick = function() {
        annotationMode = !annotationMode;
        if (annotationMode) {
          this.style.background = '#dc3545';
          this.innerHTML = '🛑 إلغاء وضع التصنيف';
          document.getElementById('counterDiv').style.display = 'block';
          alert('وضع التصنيف مفعّل!\\n\\nانقر على نقاط الحرائق لتصنيفها.\\nيمكنك تصنيف أي عدد تريده!\\nستبقى النافذة مفتوحة بعد كل تصنيف.\\n\\nعند الانتهاء، انقر على \"تحميل جميع التصنيفات\".');
        } else {
          this.style.background = '#28a745';
          this.innerHTML = '🎯 تفعيل وضع التصنيف';
          if (annotations.length === 0) {
            document.getElementById('counterDiv').style.display = 'none';
          }
        }
      };
      
      // Download annotations
      document.getElementById('downloadBtn').onclick = function() {
        if (annotations.length === 0) {
          alert('لا توجد تصنيفات للتحميل بعد!');
          return;
        }
        
        // Create CSV content
        var csv = 'latitude,longitude,label,timestamp,notes\\n';
        annotations.forEach(function(ann) {
          csv += ann.latitude + ',' + ann.longitude + ',' + ann.label + ',' + ann.timestamp + ',' + (ann.notes || '') + '\\n';
        });
        
        // Download
        var blob = new Blob([csv], { type: 'text/csv' });
        var url = window.URL.createObjectURL(blob);
        var a = document.createElement('a');
        a.href = url;
        a.download = 'fire_annotations_' + new Date().toISOString().split('T')[0] + '.csv';
        a.click();
        
        alert('✅ تم تحميل ' + annotations.length + ' تصنيف!');
      };
      
      // Create annotation dialog
      function showAnnotationDialog(latlng, fireData) {
        // Create overlay
        var overlay = document.createElement('div');
        overlay.id = 'annotationOverlay';
        overlay.style.cssText = 'position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.5);z-index:10000;display:flex;align-items:center;justify-content:center;';
        
        // Create dialog
        var dialog = document.createElement('div');
        dialog.style.cssText = 'background:white;padding:30px;border-radius:10px;box-shadow:0 4px 20px rgba(0,0,0,0.3);max-width:500px;width:90%;';
        
        dialog.innerHTML = '<h2 style=\"margin-top:0;color:#333;\">🎯 تصنيف هذا الحريق</h2>' +
          '<div style=\"background:#f8f9fa;padding:15px;border-radius:5px;margin:15px 0;\">' +
          '<p style=\"margin:5px 0;\"><strong>📍 Location:</strong> ' + latlng.lat.toFixed(4) + ', ' + latlng.lng.toFixed(4) + '</p>' +
          (fireData ? '<p style=\"margin:5px 0;\"><strong>🌍 المنطقة:</strong> ' + (fireData.region_ar || fireData.region || 'غير محدد') + '</p>' : '') +
          (fireData ? '<p style=\"margin:5px 0;\"><strong>🔥 FRP:</strong> ' + (fireData.frp ? fireData.frp.toFixed(1) : 'N/A') + ' MW</p>' : '') +
          (fireData ? '<p style=\"margin:5px 0;\"><strong>🌿 NDVI:</strong> ' + (fireData.ndvi ? fireData.ndvi.toFixed(3) : 'N/A') + '</p>' : '') +
          '</div>' +
          '<h3 style=\"color:#333;\">ما نوع هذا الحريق؟</h3>' +
          '<button id=\"flareBtn\" style=\"width:100%;padding:15px;margin:10px 0;background:#dc3545;color:white;border:none;border-radius:5px;font-size:16px;font-weight:bold;cursor:pointer;\">🏭 مصدر صناعي</button>' +
          '<button id=\"wildfireBtn\" style=\"width:100%;padding:15px;margin:10px 0;background:#ffc107;color:#333;border:none;border-radius:5px;font-size:16px;font-weight:bold;cursor:pointer;\">🔥 حريق نباتي</button>' +
          '<button id=\"skipBtn\" style=\"width:100%;padding:10px;margin:10px 0;background:#6c757d;color:white;border:none;border-radius:5px;font-size:14px;cursor:pointer;\">❓ تخطي / غير مؤكد</button>' +
          '<div style=\"margin:15px 0;\"><textarea id=\"notesInput\" placeholder=\"إضافة ملاحظات (اختياري)...\" style=\"width:100%;padding:10px;border:1px solid #ddd;border-radius:5px;resize:vertical;\" rows=\"3\"></textarea></div>';
        
        overlay.appendChild(dialog);
        document.body.appendChild(overlay);
        
        // Handle buttons
        document.getElementById('flareBtn').onclick = function() {
          saveAnnotation(latlng, 'industrial_flare', document.getElementById('notesInput').value);
          document.body.removeChild(overlay);
        };
        
        document.getElementById('wildfireBtn').onclick = function() {
          saveAnnotation(latlng, 'wildfire', document.getElementById('notesInput').value);
          document.body.removeChild(overlay);
        };
        
        document.getElementById('skipBtn').onclick = function() {
          document.body.removeChild(overlay);
        };
        
        // Close on overlay click
        overlay.onclick = function(e) {
          if (e.target === overlay) {
            document.body.removeChild(overlay);
          }
        };
      }
      
      // Update counter
      function updateCounter() {
        document.getElementById('counterNum').textContent = annotations.length;
        if (annotations.length > 0) {
          document.getElementById('downloadBtn').style.display = 'block';
          document.getElementById('counterDiv').style.display = 'block';
        }
      }
      
      // Save annotation
      function saveAnnotation(latlng, label, notes) {
        var annotation = {
          latitude: latlng.lat.toFixed(4),
          longitude: latlng.lng.toFixed(4),
          label: label,
          timestamp: new Date().toISOString(),
          notes: notes || ''
        };
        
        annotations.push(annotation);
        updateCounter();
        
        // Show brief success message (no need to click OK)
        var labelText = label === 'industrial_flare' ? '🏭 مصدر صناعي' : '🔥 حريق نباتي';
        console.log('✅ Saved:', labelText, 'at', annotation.latitude + ', ' + annotation.longitude, '| Total:', annotations.length);
        
        // Show a quick toast notification instead of alert
        var toast = document.createElement('div');
        toast.style.cssText = 'position:fixed;top:20px;right:20px;background:#28a745;color:white;padding:15px 25px;border-radius:5px;box-shadow:0 4px 12px rgba(0,0,0,0.3);z-index:10001;font-weight:bold;';
        toast.textContent = '✅ Saved! (' + annotations.length + ' total)';
        document.body.appendChild(toast);
        setTimeout(function() {
          document.body.removeChild(toast);
        }, 1500);
      }
      
      // Store reference to map
      var map = this;
      
      // Function to add click handlers to all markers
      function addMarkerClickHandlers() {
        map.eachLayer(function(layer) {
          if (layer instanceof L.CircleMarker) {
            layer.off('click'); // Remove old handlers
            layer.on('click', function(e) {
              if (annotationMode) {
                L.DomEvent.stopPropagation(e); // Prevent map click
                showAnnotationDialog(e.latlng, null);
              }
            });
          }
        });
      }
      
      // Add handlers initially
      setTimeout(addMarkerClickHandlers, 1000);
      
      // Re-add handlers when layers change
      map.on('layeradd', function() {
        setTimeout(addMarkerClickHandlers, 100);
      });
      
      // Fallback: map click for empty areas
      this.on('click', function(e) {
        if (annotationMode) {
          showAnnotationDialog(e.latlng, null);
        }
      });
    }
  ") |>
  addLayersControl(
    position = "topright",
    baseGroups    = c("Google Maps", "Google Satellite", "Google Hybrid",
                      "خريطة مبسطة","صور الأقمار","طبوغرافية"),
    overlayGroups = c("حدود المملكة","المناطق",
                      "مؤشر المخاطر الشامل (KRI)",
                      "مؤشر طقس الحريق (FWI)",
                      "جميع الكشوفات - آخر 24 ساعة",
                      "جميع الكشوفات - آخر 7 أيام",
                      "حرائق نباتية فقط - آخر 24 ساعة",
                      "حرائق نباتية فقط - آخر 7 أيام",
                      "توهجات صناعية فقط - آخر 24 ساعة",
                      "توهجات صناعية فقط - آخر 7 أيام",
                      day_groups),
    options = layersControlOptions(collapsed = TRUE)
  ) |>
  showGroup(c("جميع الكشوفات - آخر 7 أيام","حدود المملكة","المناطق",
              "مؤشر طقس الحريق (FWI)")) |>
  hideGroup(c("جميع الكشوفات - آخر 24 ساعة",
              "حرائق نباتية فقط - آخر 24 ساعة",
              "حرائق نباتية فقط - آخر 7 أيام",
              "توهجات صناعية فقط - آخر 24 ساعة",
              "توهجات صناعية فقط - آخر 7 أيام",
              day_groups,
              "مؤشر المخاطر الشامل (KRI)")) |>
  setView(lng = center[1], lat = center[2], zoom = 5)
