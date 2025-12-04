# main.py - Unified Soil Classification API (CNN + SVM + Random Forest)
import os
# Force TensorFlow to use legacy Keras (compatible with older .h5 models)
os.environ['TF_USE_LEGACY_KERAS'] = '1'

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from PIL import Image
import io
import numpy as np
import tensorflow as tf
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Optional, Literal
import logging
import pickle
import requests
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Preprocessor Class (needed for loading pickle) ===
class SoilDataPreprocessor:
    """Preprocessor for soil chemistry data"""
    def __init__(self):
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

# === Configuration ===
ATLAS_URI = "mongodb+srv://achidubem1215_db_user:batkid123@soil-cluster.oewl8ku.mongodb.net/soil_db?retryWrites=true&w=majority"

# Model paths
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

CNN_MODEL_PATH = MODEL_DIR / "soil_model_7class.h5"
SVM_MODEL_PATH = MODEL_DIR / "svm_model.pkl"
RF_MODEL_PATH = MODEL_DIR / "randomforest_model.pkl"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.pkl"

# Dropbox direct download URL (change dl=0 to dl=1 for direct download)
CNN_MODEL_URL = "https://www.dropbox.com/scl/fi/pc0dgb8u9i5okoxgpm051/soil_model_7class.h5?rlkey=ggzdkjlypiixbiiyxbud3ex0q&st=msyr0vaw&dl=1"

# === Soil Classes and Crop Suggestions ===
CLASS_NAMES = [
    'Alluvial_Soil', 'Arid_Soil', 'Black_Soil', 'Laterite_Soil',
    'Mountain_Soil', 'Red_Soil', 'Yellow_Soil'
]

CROP_SUGGESTIONS = {
    'Alluvial_Soil': 'Rice, Wheat, Sugarcane, Pulses',
    'Arid_Soil': 'Barley, Jowar, Cactus, Millets',
    'Black_Soil': 'Cotton, Soybean, Groundnut, Pulses',
    'Laterite_Soil': 'Tea, Coffee, Rubber, Cashew',
    'Mountain_Soil': 'Apples, Tea, Herbs, Spices',
    'Red_Soil': 'Groundnut, Millets, Tobacco, Potato',
    'Yellow_Soil': 'Maize, Turmeric, Sugarcane, Soybean'
}

# === Global variables ===
client = None
db = None
collection = None
cnn_model = None
svm_model = None
rf_model = None
preprocessor = None

# === Pydantic Models ===
class SoilChemistryInput(BaseModel):
    sand_percent: float = Field(..., ge=0, le=100, description="Sand percentage")
    silt_percent: float = Field(..., ge=0, le=100, description="Silt percentage")
    clay_percent: float = Field(..., ge=0, le=100, description="Clay percentage")
    ph: float = Field(..., ge=0, le=14, description="Soil pH")
    organic_content: float = Field(..., ge=0, le=100, description="Organic matter %")
    nitrogen: Optional[float] = Field(150, ge=0, description="Nitrogen (kg/ha)")
    phosphorus: Optional[float] = Field(50, ge=0, description="Phosphorus (kg/ha)")
    potassium: Optional[float] = Field(200, ge=0, description="Potassium (kg/ha)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "sand_percent": 45.5,
                "silt_percent": 35.2,
                "clay_percent": 19.3,
                "ph": 6.8,
                "organic_content": 3.5,
                "nitrogen": 180,
                "phosphorus": 45,
                "potassium": 250
            }
        }

class ClassificationResult(BaseModel):
    soilType: str
    confidence: float
    crops: str
    method: Literal["CNN", "SVM", "RandomForest"]
    id: str = Field(..., alias="_id")
    createdAt: datetime
    
    class Config:
        populate_by_name = True

class ChemistryClassificationResult(BaseModel):
    soilType: str
    confidence: float
    crops: str
    method: Literal["SVM", "RandomForest"]
    model_comparison: dict
    id: str = Field(..., alias="_id")
    createdAt: datetime
    input_data: dict
    
    class Config:
        populate_by_name = True
        protected_namespaces = ()  # Fix Pydantic warning

class HistoryItem(BaseModel):
    soilType: str
    confidence: float
    crops: str
    method: str
    id: str = Field(..., alias="_id")
    createdAt: datetime
    
    class Config:
        populate_by_name = True

class HistoryResponse(BaseModel):
    data: List[HistoryItem]
    total: int
    limit: int

class HealthResponse(BaseModel):
    status: str
    database: str
    cnn_model: str
    svm_model: str
    rf_model: str
    preprocessor: str
    timestamp: datetime

# === Helper: Download CNN model from Dropbox ===
def download_cnn_model():
    """Download CNN model from Dropbox if not present locally"""
    if CNN_MODEL_PATH.exists():
        file_size = CNN_MODEL_PATH.stat().st_size
        if file_size > 1000000:  # At least 1MB
            logger.info(f"✅ CNN model already exists locally ({file_size / 1024 / 1024:.1f} MB)")
            return
        else:
            logger.warning(f"⚠️ Existing model file is too small ({file_size} bytes), re-downloading...")
            CNN_MODEL_PATH.unlink()
    
    logger.info("📥 Downloading soil_model_7class.h5 from Dropbox (~112 MB)...")
    
    try:
        # Direct download from Dropbox (dl=1 parameter ensures direct download)
        response = requests.get(CNN_MODEL_URL, stream=True, timeout=300)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        chunk_size = 1024 * 1024  # 1MB chunks
        
        with open(CNN_MODEL_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0 and downloaded % (20 * 1024 * 1024) == 0:  # Log every 20MB
                        percent = (downloaded / total_size) * 100
                        logger.info(f"Downloaded: {downloaded / 1024 / 1024:.1f} MB ({percent:.1f}%)")
        
        # Verify file size
        final_size = CNN_MODEL_PATH.stat().st_size
        if final_size < 1000000:  # Less than 1MB
            raise Exception(f"Downloaded file is too small ({final_size} bytes)")
        
        logger.info(f"✅ CNN model downloaded successfully! ({final_size / 1024 / 1024:.1f} MB)")
        
    except Exception as e:
        logger.error(f"❌ Failed to download CNN model: {e}")
        logger.warning("⚠️ API will start without CNN model - image classification will fail")
        # Don't raise - allow API to start without model

# === Lifespan Events ===
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    global client, db, collection, cnn_model, svm_model, rf_model, preprocessor
    
    logger.info("🚀 Starting Unified Soil Classifier API...")
    
    # Connect to MongoDB
    logger.info("📡 Connecting to MongoDB Atlas...")
    try:
        client = AsyncIOMotorClient(
            ATLAS_URI,
            serverSelectionTimeoutMS=10000,
            connectTimeoutMS=20000,
            tlsAllowInvalidCertificates=True,  # Fix for Render SSL issues
            retryWrites=True,
            w='majority'
        )
        db = client["soil_db"]
        collection = db["classifications"]
        await client.admin.command('ping')
        logger.info("✅ Connected to MongoDB Atlas")
        await collection.create_index([("createdAt", -1)])
    except Exception as e:
        logger.error(f"❌ MongoDB connection failed: {e}")
        logger.warning("⚠️ Continuing without database (classification will fail)")
        # Don't raise - allow app to start even if DB fails
    
    # Download CNN model if missing
    download_cnn_model()
    
    # Load CNN Model
    logger.info(f"🤖 Loading CNN model from: {CNN_MODEL_PATH}")
    try:
        if CNN_MODEL_PATH.exists():
            # Load model with backward compatibility for older Keras formats
            import tensorflow as tf
            # Disable Keras 3 if available (TF 2.15 can use Keras 2 or 3)
            import os
            os.environ['TF_USE_LEGACY_KERAS'] = '1'
            
            cnn_model = tf.keras.models.load_model(
                str(CNN_MODEL_PATH),
                compile=False  # Skip compilation to avoid optimizer issues
            )
            logger.info("✅ CNN model loaded successfully")
        else:
            logger.warning("⚠️ CNN model file not found - image classification will be disabled")
    except Exception as e:
        logger.error(f"❌ CNN model loading failed: {e}")
        logger.warning("⚠️ Continuing without CNN model - will retry using legacy Keras")
        try:
            # Fallback: Try with h5py directly
            import h5py
            with h5py.File(str(CNN_MODEL_PATH), 'r') as f:
                logger.info("Model file is readable, trying alternative load method...")
            # Try loading with custom objects
            cnn_model = tf.keras.models.load_model(
                str(CNN_MODEL_PATH),
                compile=False,
                options=tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')
            )
            logger.info("✅ CNN model loaded with fallback method")
        except Exception as e2:
            logger.error(f"❌ Fallback loading also failed: {e2}")
            logger.warning("⚠️ API will continue without CNN model")
    
    # Load SVM Model
    logger.info(f"🤖 Loading SVM model from: {SVM_MODEL_PATH}")
    try:
        with open(SVM_MODEL_PATH, 'rb') as f:
            svm_model = pickle.load(f)
        logger.info("✅ SVM model loaded")
    except Exception as e:
        logger.warning(f"⚠️ SVM model not found: {e}")
    
    # Load Random Forest Model
    logger.info(f"🤖 Loading Random Forest model from: {RF_MODEL_PATH}")
    try:
        with open(RF_MODEL_PATH, 'rb') as f:
            rf_model = pickle.load(f)
        logger.info("✅ Random Forest model loaded")
    except Exception as e:
        logger.warning(f"⚠️ Random Forest model not found: {e}")
    
    # Load Preprocessor
    logger.info(f"🔧 Loading preprocessor from: {PREPROCESSOR_PATH}")
    try:
        # Ensure the SoilDataPreprocessor class is available in this module
        import sys
        sys.modules['__main__'].SoilDataPreprocessor = SoilDataPreprocessor
        
        with open(PREPROCESSOR_PATH, 'rb') as f:
            preprocessor = pickle.load(f)
        logger.info("✅ Preprocessor loaded")
    except Exception as e:
        logger.warning(f"⚠️ Preprocessor not found: {e}")
        logger.info("Creating new preprocessor instance...")
        preprocessor = SoilDataPreprocessor()
    
    logger.info("🎉 API is ready!")
    
    yield
    
    logger.info("🛑 Shutting down...")
    if client:
        client.close()

# === Initialize App ===
app = FastAPI(
    title="Unified Soil Classifier API",
    version="3.1",
    description="AI-powered soil classification using CNN (images) or ML models (chemistry data)",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === ENDPOINTS ===

@app.get("/", tags=["General"])
async def root():
    """Root endpoint"""
    return {
        "message": "🌱 Unified Soil Classification API",
        "version": "3.1",
        "methods": {
            "image": "CNN classification from soil images",
            "chemistry": "SVM/RF classification from soil chemistry data"
        },
        "endpoints": {
            "classify_image": "/classify (POST)",
            "classify_chemistry": "/classify-chemistry (POST)",
            "history": "/history (GET)",
            "health": "/health (GET)"
        }
    }

@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint"""
    db_status = "disconnected"
    try:
        await client.admin.command('ping')
        db_status = "connected"
    except:
        pass
    
    return HealthResponse(
        status="healthy" if db_status == "connected" else "partial",
        database=db_status,
        cnn_model="loaded" if cnn_model else "not loaded",
        svm_model="loaded" if svm_model else "not loaded",
        rf_model="loaded" if rf_model else "not loaded",
        preprocessor="loaded" if preprocessor else "not loaded",
        timestamp=datetime.utcnow()
    )

@app.post("/classify", response_model=ClassificationResult, tags=["Classification"])
async def classify_image(file: UploadFile = File(...)):
    """
    Classify soil type from an uploaded image using CNN.
    """
    if not cnn_model:
        raise HTTPException(status_code=503, detail="CNN model not available")
    
    # More lenient content type checking
    if file.content_type:
        if not (file.content_type.startswith("image/") or file.content_type == "application/octet-stream"):
            logger.warning(f"Unexpected content type: {file.content_type}, but will try to process anyway")
    
    # Check file extension as backup
    if file.filename:
        extension = file.filename.lower().split('.')[-1]
        allowed_extensions = ['jpg', 'jpeg', 'png', 'gif', 'webp', 'bmp', 'heic', 'heif']
        if extension not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid file extension: .{extension}. Allowed: {', '.join(allowed_extensions)}"
            )
    
    try:
        contents = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {str(e)}")
    
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (max 10MB)")
    
    if len(contents) < 100:
        raise HTTPException(status_code=400, detail="File too small or empty")
    
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        logger.info(f"📸 Processing image: {file.filename} ({len(contents)} bytes, {image.size})")
    except Exception as e:
        logger.error(f"Failed to open image: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
    
    # Preprocess and predict
    image = image.resize((224, 224))
    arr = np.array(image, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    
    try:
        predictions = cnn_model.predict(arr, verbose=0)
        idx = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][idx])
        soil_type = CLASS_NAMES[idx]
        crops = CROP_SUGGESTIONS.get(soil_type, "No suggestions")
        
        logger.info(f"✅ CNN Prediction: {soil_type} ({confidence:.2%})")
    except Exception as e:
        logger.error(f"Model prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    # Save to database
    try:
        result = await collection.insert_one({
            "soilType": soil_type,
            "confidence": confidence,
            "crops": crops,
            "method": "CNN",
            "createdAt": datetime.utcnow(),
            "filename": file.filename
        })
    except Exception as e:
        logger.warning(f"Failed to save to database: {e}")
        # Continue anyway - prediction succeeded
    
    return ClassificationResult(
        soilType=soil_type,
        confidence=round(confidence, 4),
        crops=crops,
        method="CNN",
        _id=str(result.inserted_id) if 'result' in locals() else "local",
        createdAt=datetime.utcnow()
    )

@app.post("/classify-chemistry", response_model=ChemistryClassificationResult, tags=["Classification"])
async def classify_chemistry(data: SoilChemistryInput):
    """
    Classify soil type from chemistry data using SVM and Random Forest.
    Returns predictions from both models for comparison.
    """
    if not svm_model or not rf_model or not preprocessor:
        raise HTTPException(
            status_code=503,
            detail="ML models or preprocessor not available. Run training script first."
        )
    
    # Validate that texture percentages sum to ~100
    texture_sum = data.sand_percent + data.silt_percent + data.clay_percent
    if not (98 <= texture_sum <= 102):
        raise HTTPException(
            status_code=400,
            detail=f"Sand + Silt + Clay must sum to 100% (got {texture_sum:.1f}%)"
        )
    
    # Prepare input data
    input_array = np.array([[
        data.sand_percent,
        data.silt_percent,
        data.clay_percent,
        data.ph,
        data.organic_content,
        data.nitrogen,
        data.phosphorus,
        data.potassium
    ]])
    
    # Normalize using saved preprocessor
    input_normalized = preprocessor.scaler.transform(input_array)
    
    # Predict with both models
    svm_pred = svm_model.predict(input_normalized)[0]
    svm_proba = svm_model.predict_proba(input_normalized)[0]
    svm_confidence = float(svm_proba[svm_pred])
    svm_soil_type = preprocessor.label_encoder.inverse_transform([svm_pred])[0]
    
    rf_pred = rf_model.predict(input_normalized)[0]
    rf_proba = rf_model.predict_proba(input_normalized)[0]
    rf_confidence = float(rf_proba[rf_pred])
    rf_soil_type = preprocessor.label_encoder.inverse_transform([rf_pred])[0]
    
    # Use the model with higher confidence
    if svm_confidence >= rf_confidence:
        final_soil_type = svm_soil_type
        final_confidence = svm_confidence
        final_method = "SVM"
    else:
        final_soil_type = rf_soil_type
        final_confidence = rf_confidence
        final_method = "RandomForest"
    
    crops = CROP_SUGGESTIONS.get(final_soil_type, "No suggestions")
    
    logger.info(f"✅ SVM: {svm_soil_type} ({svm_confidence:.2%})")
    logger.info(f"✅ RF: {rf_soil_type} ({rf_confidence:.2%})")
    logger.info(f"🎯 Final: {final_soil_type} using {final_method}")
    
    # Save to database
    result = await collection.insert_one({
        "soilType": final_soil_type,
        "confidence": final_confidence,
        "crops": crops,
        "method": final_method,
        "model_comparison": {
            "SVM": {"prediction": svm_soil_type, "confidence": svm_confidence},
            "RandomForest": {"prediction": rf_soil_type, "confidence": rf_confidence}
        },
        "input_data": data.dict(),
        "createdAt": datetime.utcnow()
    })
    
    return ChemistryClassificationResult(
        soilType=final_soil_type,
        confidence=round(final_confidence, 4),
        crops=crops,
        method=final_method,
        model_comparison={
            "SVM": {"prediction": svm_soil_type, "confidence": round(svm_confidence, 4)},
            "RandomForest": {"prediction": rf_soil_type, "confidence": round(rf_confidence, 4)}
        },
        _id=str(result.inserted_id),
        createdAt=datetime.utcnow(),
        input_data=data.dict()
    )

@app.get("/history", response_model=HistoryResponse, tags=["History"])
async def get_history(
    limit: int = 20,
    skip: int = 0,
    method: Optional[str] = None
):
    """Get classification history with optional filtering by method"""
    if limit > 100:
        limit = 100
    
    try:
        query = {}
        if method:
            query["method"] = method
        
        total = await collection.count_documents(query)
        cursor = collection.find(query).sort("createdAt", -1).skip(skip).limit(limit)
        results = await cursor.to_list(length=limit)
        
        data = [
            HistoryItem(
                soilType=r["soilType"],
                confidence=r["confidence"],
                crops=r.get("crops", ""),
                method=r.get("method", "CNN"),
                _id=str(r["_id"]),
                createdAt=r["createdAt"]
            )
            for r in results
        ]
        
        return HistoryResponse(data=data, total=total, limit=limit)
    
    except Exception as e:
        logger.error(f"Failed to retrieve history: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve history")

@app.delete("/history/{record_id}", tags=["History"])
async def delete_record(record_id: str):
    """Delete a specific classification record"""
    from bson import ObjectId
    
    try:
        result = await collection.delete_one({"_id": ObjectId(record_id)})
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Record not found")
        return {"message": "Record deleted", "id": record_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
