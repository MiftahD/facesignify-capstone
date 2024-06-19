const crypto = require('crypto');
const tf = require('@tensorflow/tfjs-node');
const sharp = require('sharp');
const { IMAGENET_CLASSES } = require('../config/imagenet_classes');
const { admin, db } = require('../config/firebaseConfig');
const { Storage } = require('@google-cloud/storage');
require('dotenv').config();

const storage = new Storage();
const bucket = storage.bucket(process.env.GCLOUD_STORAGE_BUCKET);

async function loadModel() {
  try {
    const model = await tf.loadGraphModel(process.env.MODEL_URL);
    return model;
  } catch (error) {
    console.error('Error loading model:', error);
    throw new Error('Failed to load model');
  }
}

const predict = async (imageBuffer) => {
  const model = await loadModel();
  const fileName = crypto.randomUUID();
  const filePath = `images/${fileName}.jpeg`;

  try {
    // Upload image to Cloud Storage
    const file = bucket.file(filePath);
    await new Promise((resolve, reject) => {
      const stream = file.createWriteStream({
        metadata: {
          contentType: 'image/jpeg',
        },
      });

      stream.on('error', reject);
      stream.on('finish', resolve);
      stream.end(imageBuffer);
    });

    const publicUrl = `https://storage.googleapis.com/${bucket.name}/${filePath}`;
    
    // Resize and process image for prediction
    const processedImageBuffer = await sharp(imageBuffer)
      .resize(160, 160)
      .toFormat('jpeg')
      .toBuffer();

    const imageTensor = tf.node.decodeImage(processedImageBuffer, 3)
      .expandDims(0)
      .toFloat()
      .div(tf.scalar(255));

    console.log('Image tensor shape:', imageTensor.shape);

    // Perform prediction using TensorFlow model
    const predictions = await model.predict(imageTensor).data();

    const result = Array.from(predictions)
      .map((p, i) => ({
        probability: p,
        className: IMAGENET_CLASSES[i],
      }))
      .sort((a, b) => b.probability - a.probability)
      .slice(0, 1);

    // Save prediction result to Firestore
    await savePredictionToFirestore(result, publicUrl);

    return result;
  } catch (error) {
    console.error('Error during prediction:', error);
    throw new Error('Failed to process image: ' + error.message);
  }
};

const savePredictionToFirestore = async (result, publicUrl) => {
  try {
    const predictionData = {
      result,
      publicUrl,
      createdAt: admin.firestore.FieldValue.serverTimestamp(),
    };
    await db.collection('predictions').add(predictionData);
    console.log('Prediction saved to Firestore successfully.');
  } catch (error) {
    console.error('Error saving prediction to Firestore:', error);
    throw new Error('Failed to save prediction to Firestore: ' + error.message);
  }
};

module.exports = { predict };
