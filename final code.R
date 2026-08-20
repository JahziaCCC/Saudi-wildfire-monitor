# ==============================================================
# SAUDI ARABIA WILDFIRE & INDUSTRIAL RISK MONITOR
# ==============================================================

# 1) PACKAGES & SETUP
required_packages <- c("sf", "leaflet", "dplyr", "readr", "httr2", "jsonlite", "htmlwidgets", "fs")
new_packages <- required_packages[!(required_packages %in% installed.packages()[,"Package"])]
if(length(new_packages)) install.packages(new_packages, repos = "https://cloud.r-project.org")

library(sf)
library(leaflet)
library(dplyr)
library(readr)
library(httr2)
library(jsonlite)
library(htmlwidgets)

# Create output directory
if(!dir.exists("./output")) dir.create("./output")

# 2) DATA RETRIEVAL (NASA FIRMS)
firms_key <- Sys.getenv("FIRMS_MAP_KEY")
if (firms_key == "") firms_key <- "OPEN_DATA_KEY" 

sources <- c("VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT", "MODIS_NRT")
all_detections <- list()

for (src in sources) {
  # استخدام المفتاح الممرر من البيئة بشكل صحيح في الرابط
  url <- sprintf("https://firms.modaps.eosdis.nasa.gov/api/country/csv/%s/%s/SAU/1", firms_key, src)
  tryCatch({
    res <- read_csv(url, show_col_types = FALSE)
    if (nrow(res) > 0) {
      res$source_sensor <- src
      all_detections[[src]] <- res
    }
  }, error = function(e) {
    cat(sprintf("⚠️ Notice: Could not fetch data for source %s\n", src))
  })
}

raw_data <- bind_rows(all_detections)

# FUNCTIONS FOR EXPORT & SUMMARY
generate_fire_summary <- function(sf_data) {
  df <- sf_data %>% st_drop_geometry()
  cat("\n==========================================")
  cat("\n   SAUDI WILDFIRE RISK SUMMARY REPORT     ")
  cat("\n==========================================\n")
  cat(sprintf("Total Active Detections : %d\n", nrow(df)))
  cat(sprintf("Classified Wildfires    : %d\n", sum(df$ml_pred_class == "wildfire", na.rm = TRUE)))
  cat(sprintf("Industrial Flares       : %d\n", sum(df$ml_pred_class == "flare", na.rm = TRUE)))
  cat(sprintf("High Risk Wildfires     : %d (KRI >= 60)\n", sum(df$ml_pred_class == "wildfire" & df$kri_score >= 60, na.rm = TRUE)))
  cat("------------------------------------------\n")
}

export_fire_data <- function(sf_data) {
  date_str <- format(Sys.Date(), "%Y%m%d")
  csv_file <- sprintf("./output/saudi_wildfire_analysis_%s.csv", date_str)
  geojson_file <- sprintf("./output/wildfires_only_%s.geojson", date_str)
  
  df_export <- sf_data %>% 
    mutate(longitude = st_coordinates(.)[,1], latitude = st_coordinates(.)[,2]) %>% 
    st_drop_geometry()
  write_csv(df_export, csv_file)
  cat(sprintf("✓ CSV exported successfully: %s\n", csv_file))
  
  wildfires_only <- sf_data %>% filter(ml_pred_class == "wildfire")
  if (nrow(wildfires_only) > 0) {
    st_write(wildfires_only, geojson_file, delete_dsn = TRUE, quiet = TRUE)
    cat(sprintf("✓ GeoJSON exported successfully: %s\n", geojson_file))
  }
}

export_interactive_map <- function(sf_data) {
  date_str <- format(Sys.Date(), "%Y%m%d")
  map_file <- sprintf("./output/fire_map_%s.html", date_str)
  
  m <- leaflet(sf_data) %>%
    addProviderTiles(providers$CartoDB.Positron) %>%
    addCircleMarkers(
      radius = ~ifelse(ml_pred_class == "wildfire", 6, 4),
      color = ~ifelse(ml_pred_class == "wildfire", "#E41A1C", "#377EB8"),
      stroke = FALSE, 
      fillOpacity = 0.8,
      popup = ~paste0(
        "<b>النوع:</b> ", ml_pred_class, "<br>",
        "<b>درجة الخطورة (KRI):</b> ", kri_score, "<br>",
        "<b>مؤشر طقس الحرائق (FWI):</b> ", fwi_score
      )
    )
  
  saveWidget(m, file = map_file, selfcontained = TRUE)
  cat(sprintf("✓ Interactive map exported: %s\n", map_file))
  return(map_file)
}

send_telegram_alert <- function(token, chat_id, message_text, files = NULL) {
  url_msg <- paste0("https://api.telegram.org/bot", token, "/sendMessage")
  req_msg <- request(url_msg) |>
    req_body_json(list(chat_id = chat_id, text = message_text, parse_mode = "Markdown"))
  tryCatch(req_perform(req_msg), error = function(e) cat("⚠️ خطأ في إرسال الرسالة\n"))
  
  if (!is.null(files)) {
    for (file_path in files) {
      if (file.exists(file_path)) {
        url_doc <- paste0("https://api.telegram.org/bot", token, "/sendDocument")
        req_doc <- request(url_doc) |>
          req_body_multipart(chat_id = chat_id, document = curl::form_file(file_path))
        tryCatch(req_perform(req_doc), error = function(e) cat("⚠️ خطأ في إرسال الملف:", file_path, "\n"))
      }
    }
  }
}

# 3) PROCESSING & TELEGRAM NOTIFICATION
if (nrow(raw_data) > 0) {
  fires_sf <- st_as_sf(raw_data, coords = c("longitude", "latitude"), crs = 4326)
  
  fires_sf <- fires_sf %>%
    mutate(
      confidence_num = suppressWarnings(as.numeric(confidence)),
      confidence_num = ifelse(is.na(confidence_num), 50, confidence_num),
      ml_pred_class = ifelse(confidence_num >= 85 & bright_ti4 > 340, "flare", "wildfire"),
      fwi_score = round((frp * 0.4) + (bright_ti4 * 0.1) - 25, 1),
      kri_score = round(pmin(pmax((fwi_score * 1.2) + (confidence_num * 0.3), 0), 100), 1)
    )
    
  generate_fire_summary(fires_sf)
  export_fire_data(fires_sf)
  latest_map <- export_interactive_map(fires_sf)
  latest_csv <- file.path("./output", paste0("saudi_wildfire_analysis_", format(Sys.Date(), "%Y%m%d"), ".csv"))
  
  df_summary <- fires_sf |> st_drop_geometry()
  report_text <- paste0(
    "🚨 *تنبيه نظام مراقبة الحرائق - السعودية*\n\n",
    "📊 *ملخص التقرير اليومي:*\n",
    "• إجمالي النقاط المكتشفة: *", nrow(df_summary), "*\n",
    "• الحرائق الطبيعية: *", sum(df_summary$ml_pred_class == "wildfire", na.rm = TRUE), "*\n",
    "• التوهجات الصناعية: *", sum(df_summary$ml_pred_class == "flare", na.rm = TRUE), "*\n",
    "• الحرائق عالية الخطورة (KRI ≥ 60): *", sum(df_summary$ml_pred_class == "wildfire" & df_summary$kri_score >= 60, na.rm = TRUE), "*\n\n",
    "📁 *مرفق ملف التقرير التفصيلي (CSV) والخريطة التفاعلية (HTML).*"
  )
  files_to_send <- c(latest_csv, latest_map)

} else {
  cat("\n==========================================")
  cat("\n No active detections found for today.   ")
  cat("\n==========================================\n")
  report_text <- "✅ *تقرير نظام مراقبة الحرائق - السعودية*\n\nلا توجد أي حرائق أو مؤشرات خطورة مكتشفة اليوم."
  files_to_send <- NULL
}

# SEND TELEGRAM ALERT
send_telegram_alert(
  token = Sys.getenv("TELEGRAM_TOKEN"),
  chat_id = Sys.getenv("TELEGRAM_CHAT_ID"),
  message_text = report_text,
  files = files_to_send
)
