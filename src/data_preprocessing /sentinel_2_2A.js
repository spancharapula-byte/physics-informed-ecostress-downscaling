// Our Region of Interest is California (region around Fresno)
var point = ee.Geometry.Point([-119.7871, 36.7378]);
var roi = ee.Geometry(point).buffer(15000); 

function maskS2clouds(image) {
  var img = ee.Image(image); 
  var qa = img.select('QA60');
  var cloudBitMask = 1 << 10;
  var cirrusBitMask = 1 << 11;
  var mask = qa.bitwiseAnd(cloudBitMask).eq(0)
      .and(qa.bitwiseAnd(cirrusBitMask).eq(0));
  return img.updateMask(mask).divide(10000);
}

function processAndExport(startDate, endDate, exportName) {
  var dataset = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                  .filterBounds(roi)
                  .filterDate(startDate, endDate)
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
                  .map(maskS2clouds);

  var s2Image = dataset.median().clip(roi);

  var ndvi = s2Image.normalizedDifference(['B8', 'B4']).rename('NDVI');
  var ndbi = s2Image.normalizedDifference(['B11', 'B8']).rename('NDBI');

  var finalFeatures = s2Image.select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12'])
                             .addBands(ndvi)
                             .addBands(ndbi);

  Export.image.toDrive({
    image: finalFeatures,
    description: exportName,
    folder: 'PINN_LST_Project_Data', 
    scale: 10,                       
    region: roi,
    maxPixels: 1e10                  
  });
}

processAndExport('2025-09-14', '2025-09-24', 'Sentinel2_Features_Sep19');
processAndExport('2025-09-11', '2025-09-21', 'Sentinel2_Features_Sep16');
processAndExport('2025-09-03', '2025-09-13', 'Sentinel2_Features_Sep08');
processAndExport('2025-08-28', '2025-09-07', 'Sentinel2_Features_Sep02');
processAndExport('2025-08-07', '2025-08-17', 'Sentinel2_Features_Aug12');

var visDataset = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                  .filterBounds(roi).filterDate('2025-09-14', '2025-09-24').map(maskS2clouds).median().clip(roi);
var visNdvi = visDataset.normalizedDifference(['B8', 'B4']);
Map.setCenter(-119.7871, 36.7378, 11);
Map.addLayer(visNdvi, {min: -0.2, max: 0.8, palette: ['red', 'white', 'green']}, 'NDVI Display (Sep 19 Window)');
