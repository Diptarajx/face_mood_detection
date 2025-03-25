#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <chrono>
#include <map>
#include <vector>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <ctime>
#include <filesystem>

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

// Configuration structure to store application settings
struct Config {
    int camera_id = 0;
    double scale_factor = 0.5;
    bool display_landmarks = false;
    bool save_detections = false;
    string save_directory = "detections";
    bool debug_mode = false;
    bool record_video = false;
    string video_output = "output.mp4";
    double detection_confidence = 4.0; // Min neighbors for face detection
    Size min_face_size = Size(30, 30);
    bool process_every_frame = false;
    bool show_fps = true;
    string emotion_model_path = "emotion_ferplus_v1.onnx"; // Path to emotion recognition model
    bool use_ml_emotion = true; // Flag to use ML-based emotion detection
};

// Logger class for professional logging
class Logger {
private:
    ofstream log_file;
    bool debug_enabled;
    
    string getTimestamp() {
        auto now = chrono::system_clock::now();
        auto time = chrono::system_clock::to_time_t(now);
        stringstream ss;
        ss << put_time(localtime(&time), "%Y-%m-%d %H:%M:%S");
        return ss.str();
    }
    
public:
    Logger(const string& filename = "face_detection.log", bool debug = false) : debug_enabled(debug) {
        log_file.open(filename, ios::app);
        if (!log_file.is_open()) {
            cerr << "Could not open log file: " << filename << endl;
        }
        info("Logger initialized");
    }
    
    ~Logger() {
        if (log_file.is_open()) {
            log_file.close();
        }
    }
    
    void info(const string& message) {
        string log_message = getTimestamp() + " [INFO] " + message;
        cout << log_message << endl;
        if (log_file.is_open()) {
            log_file << log_message << endl;
        }
    }
    
    void error(const string& message) {
        string log_message = getTimestamp() + " [ERROR] " + message;
        cerr << log_message << endl;
        if (log_file.is_open()) {
            log_file << log_message << endl;
        }
    }
    
    void debug(const string& message) {
        if (debug_enabled) {
            string log_message = getTimestamp() + " [DEBUG] " + message;
            cout << log_message << endl;
            if (log_file.is_open()) {
                log_file << log_message << endl;
            }
        }
    }
};

// Advanced mood detection using face features
class MoodDetector {
private:
    // HOG descriptor for feature extraction
    HOGDescriptor hog;
    // DNN-based face emotion recognition model
    dnn::Net emotion_net;
    bool model_loaded;
    
    // Global emotion state with temporal filtering
    string current_emotion;
    int emotion_stability;
    // Color map for visualization
    map<string, Scalar> emotion_colors;
    // Logger reference
    Logger& logger;
    // Debug visualization
    bool debug_mode;
    
    // Confidence scores for each emotion
    map<string, float> emotion_confidence;
    float current_confidence;
    
    // Emotion labels for the model output
    vector<string> emotion_labels;
    
    // Flag to use ML-based emotion detection
    bool use_ml;

    // Static flag to track if model loading has been attempted to avoid repeated failures
    static bool model_load_attempted;

    // Calibration data for reference emotions
    struct CalibrationData {
        bool calibrated = false;
        map<string, vector<float>> reference_metrics;
        map<string, float> reference_weights;
    };
    
    map<string, CalibrationData> calibration_data;
    bool using_calibration = false;

public:
    MoodDetector(Logger& log, bool debug = false, const string& model_path = "emotion_ferplus_v1.onnx", bool use_ml_emotion = true) 
        : logger(log), debug_mode(debug), model_loaded(false), use_ml(use_ml_emotion) {
        
        // Initialize HOG parameters optimized for faces
        hog.winSize = Size(64, 64);
        hog.blockSize = Size(16, 16);
        hog.blockStride = Size(8, 8);
        hog.cellSize = Size(8, 8);
        hog.nbins = 9;
        
        // Initialize emotion labels based on the FER+ dataset
        emotion_labels = {"Neutral", "Happy", "Surprised", "Sad", "Angry", "Disgusted", "Fearful", "Contempt"};
        
        // Initialize emotion stability
        current_emotion = "Neutral";
        emotion_stability = 0;
        current_confidence = 0.0f;
        
        // Setup emotion colors
        emotion_colors["Neutral"] = Scalar(255, 255, 255);   // White
        emotion_colors["Happy"] = Scalar(0, 255, 255);       // Yellow
        emotion_colors["Sad"] = Scalar(255, 0, 0);           // Blue
        emotion_colors["Angry"] = Scalar(0, 0, 255);         // Red
        emotion_colors["Surprised"] = Scalar(255, 0, 255);   // Purple
        emotion_colors["Disgusted"] = Scalar(0, 128, 255);   // Orange
        emotion_colors["Fearful"] = Scalar(70, 130, 180);    // Steel Blue
        emotion_colors["Contempt"] = Scalar(128, 0, 128);    // Purple
        emotion_colors["Tired"] = Scalar(128, 128, 128);     // Gray
        
        // Initialize confidence scores for all emotions
        for (const auto& label : emotion_labels) {
            emotion_confidence[label] = 0.0f;
            calibration_data[label] = CalibrationData();
        }
        emotion_confidence["Tired"] = 0.0f; // Additional emotion not in model
        calibration_data["Tired"] = CalibrationData();
        
        // Try to load the DNN model if ML emotion detection is enabled and not previously attempted
        if (use_ml && !model_load_attempted) {
            model_load_attempted = true; // Mark as attempted
            try {
                logger.info("Attempting to load emotion recognition model from: " + model_path);
                emotion_net = dnn::readNetFromONNX(model_path);
                
                // Check if the model is empty
                if (emotion_net.empty()) {
                    logger.error("Model loaded but is empty. Falling back to rule-based detection.");
                    model_loaded = false;
                } else {
                    // Set computation backend and target
                    emotion_net.setPreferableBackend(dnn::DNN_BACKEND_DEFAULT);
                    emotion_net.setPreferableTarget(dnn::DNN_TARGET_CPU);
                    model_loaded = true;
                    logger.info("Emotion recognition model loaded successfully from: " + model_path);
                }
            } catch (const cv::Exception& e) {
                logger.error("Failed to load emotion recognition model: " + string(e.what()));
                logger.error("Error details: " + string(e.err));
                logger.info("Falling back to rule-based emotion detection");
                model_loaded = false;
            }
        } else if (!use_ml) {
            logger.info("Using rule-based emotion detection (ML disabled)");
        } else if (model_load_attempted) {
            logger.info("Using rule-based emotion detection (previous model loading attempts failed)");
        }
        
        logger.info("MoodDetector initialized. ML model loaded: " + string(model_loaded ? "yes" : "no"));
    }
    
    // New method to calibrate for a specific emotion
    bool calibrateEmotion(const Mat& face_region, const string& target_emotion) {
        if (target_emotion.empty() || find(emotion_labels.begin(), emotion_labels.end(), target_emotion) == emotion_labels.end()
            && target_emotion != "Tired") {
            logger.error("Invalid emotion for calibration: " + target_emotion);
            return false;
        }
        
        // Resize and convert to grayscale
        Mat face_resized, gray_face;
        resize(face_region, face_resized, Size(64, 64));
        if (face_resized.channels() == 3) {
            cvtColor(face_resized, gray_face, COLOR_BGR2GRAY);
        } else {
            gray_face = face_resized.clone();
        }
        equalizeHist(gray_face, gray_face);
        
        // Extract facial metrics
        vector<float> metrics = extractFacialMetrics(gray_face);
        
        // Store calibration data
        if (calibration_data[target_emotion].reference_metrics.empty()) {
            // Initialize with weights for each metric
            calibration_data[target_emotion].reference_weights["smile_indicator"] = 2.0f;
            calibration_data[target_emotion].reference_weights["gradient_smile"] = 1.0f;
            calibration_data[target_emotion].reference_weights["mouth_stddev"] = 1.5f;
            calibration_data[target_emotion].reference_weights["eye_left_stddev"] = 1.0f;
            calibration_data[target_emotion].reference_weights["eye_right_stddev"] = 1.0f;
            calibration_data[target_emotion].reference_weights["forehead_stddev"] = 1.0f;
            calibration_data[target_emotion].reference_weights["face_symmetry"] = 1.5f;
            calibration_data[target_emotion].reference_weights["eye_openness"] = 1.0f;
        }
        
        // Store the metrics
        calibration_data[target_emotion].reference_metrics["facial_metrics"] = metrics;
        calibration_data[target_emotion].calibrated = true;
        using_calibration = true;
        
        logger.info("Calibrated for emotion: " + target_emotion);
        if (debug_mode) {
            string metrics_str = "Calibration metrics for " + target_emotion + ": ";
            for (size_t i = 0; i < metrics.size(); i++) {
                metrics_str += to_string(metrics[i]) + " ";
            }
            logger.debug(metrics_str);
        }
        
        return true;
    }
    
    // Helper method to extract facial metrics
    vector<float> extractFacialMetrics(const Mat& gray_face) {
        vector<float> metrics;
        
        // Calculate regions
        Mat mouth_region = gray_face(Rect(16, 40, 32, 16));
        Mat left_eye_region = gray_face(Rect(12, 16, 16, 8));
        Mat right_eye_region = gray_face(Rect(36, 16, 16, 8));
        Mat forehead_region = gray_face(Rect(16, 4, 32, 12));
        
        // Calculate statistics
        Scalar mean_mouth, stddev_mouth;
        meanStdDev(mouth_region, mean_mouth, stddev_mouth);
        
        Scalar mean_eye_left, stddev_eye_left;
        meanStdDev(left_eye_region, mean_eye_left, stddev_eye_left);
        
        Scalar mean_eye_right, stddev_eye_right;
        meanStdDev(right_eye_region, mean_eye_right, stddev_eye_right);
        
        Scalar mean_forehead, stddev_forehead;
        meanStdDev(forehead_region, mean_forehead, stddev_forehead);
        
        // Face symmetry
        Mat left_half = gray_face(Rect(0, 0, 32, 64));
        Mat right_half = gray_face(Rect(32, 0, 32, 64));
        
        Scalar mean_left, stddev_left;
        meanStdDev(left_half, mean_left, stddev_left);
        
        Scalar mean_right, stddev_right;
        meanStdDev(right_half, mean_right, stddev_right);
        
        float face_symmetry = abs(mean_left[0] - mean_right[0]);
        
        // Smile indicator
        Mat upper_mouth = mouth_region(Rect(8, 0, 16, 8));
        Mat lower_mouth = mouth_region(Rect(8, 8, 16, 8));
        
        Scalar mean_upper, stddev_upper;
        meanStdDev(upper_mouth, mean_upper, stddev_upper);
        
        Scalar mean_lower, stddev_lower;
        meanStdDev(lower_mouth, mean_lower, stddev_lower);
        
        float smile_indicator = mean_lower[0] - mean_upper[0] + (stddev_lower[0] - stddev_upper[0]) * 2.0f;
        
        // Gradient smile
        Mat mouth_sobel_y;
        Sobel(mouth_region, mouth_sobel_y, CV_32F, 0, 1);
        Mat smile_curve;
        cv::reduce(mouth_sobel_y, smile_curve, 1, cv::REDUCE_AVG);
        float gradient_smile = 0.0f;
        if (smile_curve.rows > 2) {
            gradient_smile = smile_curve.at<float>(0) - smile_curve.at<float>(smile_curve.rows - 1);
        }
        
        // Eye openness
        float eye_left_openness = stddev_eye_left[0] / mean_eye_left[0];
        float eye_right_openness = stddev_eye_right[0] / mean_eye_right[0];
        float eye_openness = (eye_left_openness + eye_right_openness) / 2.0f;
        
        // Store metrics
        metrics.push_back(smile_indicator);        // 0
        metrics.push_back(gradient_smile);         // 1
        metrics.push_back(stddev_mouth[0]);        // 2
        metrics.push_back(stddev_eye_left[0]);     // 3
        metrics.push_back(stddev_eye_right[0]);    // 4
        metrics.push_back(stddev_forehead[0]);     // 5
        metrics.push_back(face_symmetry);          // 6
        metrics.push_back(eye_openness);           // 7
        
        return metrics;
    }
    
    string detectMood(const Mat& face_region, Mat& debug_output) {
        // Resize face region to standard size
        Mat face_resized;
        resize(face_region, face_resized, Size(64, 64));
        
        // Convert to grayscale
        Mat gray_face;
        if (face_resized.channels() == 3) {
            cvtColor(face_resized, gray_face, COLOR_BGR2GRAY);
        } else {
            gray_face = face_resized.clone();
        }
        
        // Normalize lighting
        equalizeHist(gray_face, gray_face);
        
        // Reset confidence scores
        for (auto& conf : emotion_confidence) {
            conf.second = 0.0f;
        }
        
        string detected_emotion;
        
        // Use DNN model if available and ML is enabled
        if (model_loaded && use_ml) {
            if (debug_mode) {
                logger.debug("Using ML-based emotion detection");
            }
            
            try {
                // Preprocess the image for the model 
                // FER+ model expects 64x64 grayscale images with values in [0,1]
                cv::Mat inputBlob;
                
                // Resize to 64x64 if not already that size
                if (gray_face.size() != Size(64, 64)) {
                    resize(gray_face, gray_face, Size(64, 64));
                }
                
                // Convert to float and normalize to [0,1]
                Mat floatImage;
                gray_face.convertTo(floatImage, CV_32F, 1.0/255.0);
                
                // Create blob from image with correct dimensions
                // NCHW format: (batch_size, channels, height, width)
                inputBlob = dnn::blobFromImage(floatImage, 1.0, Size(64, 64), 
                                              Scalar(0), false, false);
                
                // Set the input to the network
                emotion_net.setInput(inputBlob);
                
                // Forward pass to get output
                Mat prob = emotion_net.forward();
                
                if (debug_mode) {
                    logger.debug("ML emotion inference output shape: " + 
                              to_string(prob.dims) + " dims, " + 
                              (prob.dims > 0 ? to_string(prob.size[0]) : "?") + " x " + 
                              (prob.dims > 1 ? to_string(prob.size[1]) : "?") + " x " + 
                              (prob.dims > 2 ? to_string(prob.size[2]) : "?") + " x " + 
                              (prob.dims > 3 ? to_string(prob.size[3]) : "?"));
                }
                
                // Find the emotion with highest probability
                Point maxLoc;
                double maxVal;
                
                // Handle different output formats
                if (prob.dims == 4) {
                    // Some ONNX models output in NCHW format
                    // Reshape to (1, num_classes)
                    prob = prob.reshape(1, prob.total() / prob.size[3]);
                } else if (prob.dims == 2) {
                    // Already in correct format (N, C)
                    prob = prob.reshape(1, prob.size[1]);
                } else {
                    // Attempt to flatten the output
                    prob = prob.reshape(1, prob.total());
                }
                
                minMaxLoc(prob, nullptr, &maxVal, nullptr, &maxLoc);
                int class_id = maxLoc.x;
                
                if (debug_mode) {
                    logger.debug("ML emotion inference complete. Max class ID: " + to_string(class_id) + 
                               " out of " + to_string(emotion_labels.size()) + " classes");
                    
                    // Format prob values as a string for debugging
                    string probValues = "";
                    for (int i = 0; i < min(prob.cols, 10); i++) {
                        probValues += to_string(prob.at<float>(0, i)) + " ";
                    }
                    logger.debug("Output values: " + probValues);
                }
                
                if (class_id < emotion_labels.size()) {
                    detected_emotion = emotion_labels[class_id];
                    current_confidence = static_cast<float>(maxVal);
                    
                    // Set confidence scores for all emotions
                    for (int i = 0; i < emotion_labels.size() && i < prob.cols; i++) {
                        emotion_confidence[emotion_labels[i]] = prob.at<float>(0, i);
                    }
                } else {
                    // Fallback if class_id is out of range
                    logger.error("ML emotion class_id " + to_string(class_id) + " out of range, falling back to traditional method");
                    detected_emotion = detectMoodTraditional(gray_face);
                }
            } catch (const cv::Exception& e) {
                logger.error("Error during ML inference: " + string(e.what()) + ", falling back to traditional method");
                detected_emotion = detectMoodTraditional(gray_face);
            }
            
            // Debug output for DNN result
            if (debug_mode) {
                string conf_str = "";
                for (const auto& conf : emotion_confidence) {
                    conf_str += conf.first + ": " + to_string(conf.second) + " ";
                }
                logger.debug("DNN Confidence scores: " + conf_str);
            }
        } else {
            // Fallback to traditional method if model isn't available or ML is disabled
            if (debug_mode) {
                if (!model_loaded) {
                    logger.debug("Using rule-based emotion detection (ML model not loaded)");
                } else if (!use_ml) {
                    logger.debug("Using rule-based emotion detection (ML disabled by user)");
                }
            }
            detected_emotion = detectMoodTraditional(gray_face);
        }
        
        // Apply calibration if available
        if (using_calibration) {
            detected_emotion = applyCalibration(gray_face, detected_emotion);
        }
        
        // Draw emotion on debug output
        if (!debug_output.empty()) {
            const Scalar& color = emotion_colors[detected_emotion];
            putText(debug_output, detected_emotion + " (" + to_string(current_confidence) + ")", 
                    Point(10, 30), FONT_HERSHEY_SIMPLEX, 1.0, color, 2);
            
            // Add calibration status
            if (using_calibration) {
                putText(debug_output, "Calibrated", Point(10, 60), 
                       FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);
            }
        }
        
        // Update emotion stability with weight decay
        if (detected_emotion == current_emotion) {
            emotion_stability = min(10, emotion_stability + 1);
        } else {
            emotion_stability = max(0, emotion_stability - 2);
            
            // Switch emotion only if stability is low enough
            if (emotion_stability <= 0) {
                current_emotion = detected_emotion;
                emotion_stability = 1;
            } else {
                // Keep the current emotion but reduce confidence
                detected_emotion = current_emotion;
                current_confidence *= 0.9f;
            }
        }
        
        return detected_emotion;
    }
    
    // Apply calibration to emotion detection
    string applyCalibration(const Mat& gray_face, const string& detected_emotion) {
        // Extract current facial metrics
        vector<float> current_metrics = extractFacialMetrics(gray_face);
        
        // Map to store similarity scores for each calibrated emotion
        map<string, float> calibration_scores;
        float max_score = 0.0f;
        string best_emotion = detected_emotion;
        
        // Compare current metrics with calibrated reference emotions
        for (const auto& cal_pair : calibration_data) {
            const string& emotion = cal_pair.first;
            const CalibrationData& cal_data = cal_pair.second;
            
            if (!cal_data.calibrated) continue;
            
            float similarity_score = 0.0f;
            
            // Compare metrics using weighted Euclidean distance
            const vector<float>& ref_metrics = cal_data.reference_metrics.at("facial_metrics");
            float weighted_distance = 0.0f;
            float weight_sum = 0.0f;
            
            // Ensure we have metrics to compare
            if (ref_metrics.size() >= 8 && current_metrics.size() >= 8) {
                // Smile indicator (weight 2.0)
                float weight = cal_data.reference_weights.at("smile_indicator");
                weighted_distance += weight * pow(current_metrics[0] - ref_metrics[0], 2);
                weight_sum += weight;
                
                // Gradient smile (weight 1.0)
                weight = cal_data.reference_weights.at("gradient_smile");
                weighted_distance += weight * pow(current_metrics[1] - ref_metrics[1], 2);
                weight_sum += weight;
                
                // Mouth stddev (weight 1.5)
                weight = cal_data.reference_weights.at("mouth_stddev");
                weighted_distance += weight * pow(current_metrics[2] - ref_metrics[2], 2);
                weight_sum += weight;
                
                // Eye left stddev (weight 1.0)
                weight = cal_data.reference_weights.at("eye_left_stddev");
                weighted_distance += weight * pow(current_metrics[3] - ref_metrics[3], 2);
                weight_sum += weight;
                
                // Eye right stddev (weight 1.0)
                weight = cal_data.reference_weights.at("eye_right_stddev");
                weighted_distance += weight * pow(current_metrics[4] - ref_metrics[4], 2);
                weight_sum += weight;
                
                // Forehead stddev (weight 1.0)
                weight = cal_data.reference_weights.at("forehead_stddev");
                weighted_distance += weight * pow(current_metrics[5] - ref_metrics[5], 2);
                weight_sum += weight;
                
                // Face symmetry (weight 1.5)
                weight = cal_data.reference_weights.at("face_symmetry");
                weighted_distance += weight * pow(current_metrics[6] - ref_metrics[6], 2);
                weight_sum += weight;
                
                // Eye openness (weight 1.0)
                weight = cal_data.reference_weights.at("eye_openness");
                weighted_distance += weight * pow(current_metrics[7] - ref_metrics[7], 2);
                weight_sum += weight;
                
                // Normalize by weight sum
                if (weight_sum > 0) {
                    weighted_distance = sqrt(weighted_distance / weight_sum);
                }
                
                // Convert distance to similarity (higher is better)
                // Using a Gaussian-like function where similarity decreases with distance
                similarity_score = exp(-weighted_distance / 50.0f);  // 50.0f is a scaling factor
                
                // Store the score
                calibration_scores[emotion] = similarity_score;
                
                // Update best match if this is better
                if (similarity_score > max_score) {
                    max_score = similarity_score;
                    best_emotion = emotion;
                }
            }
        }
        
        // Apply calibration only if confidence is high enough
        if (max_score > 0.7f) {
            // Boost confidence for the calibrated emotion
            emotion_confidence[best_emotion] = max(emotion_confidence[best_emotion], max_score);
            current_confidence = max_score;
            
            if (debug_mode) {
                logger.debug("Calibration applied: " + detected_emotion + " -> " + best_emotion + 
                           " (score: " + to_string(max_score) + ")");
            }
            
            return best_emotion;
        }
        
        // Otherwise, stick with the original detection
        return detected_emotion;
    }
    
    // Reset calibration data
    void resetCalibration() {
        for (auto& cal_pair : calibration_data) {
            cal_pair.second.calibrated = false;
            cal_pair.second.reference_metrics.clear();
        }
        using_calibration = false;
        logger.info("Emotion calibration reset");
    }
    
    // Check if calibration is being used
    bool isUsingCalibration() const {
        return using_calibration;
    }
    
    string detectMoodTraditional(const Mat& gray_face) {
        // Extract facial features
        vector<float> features;
        vector<float> hog_features;
        hog.compute(gray_face, hog_features);
        
        // Calculate mean and standard deviation of pixel values in different face regions
        Mat mouth_region = gray_face(Rect(16, 40, 32, 16));  // Lower part of face
        Mat left_eye_region = gray_face(Rect(12, 16, 16, 8));  // Left eye region
        Mat right_eye_region = gray_face(Rect(36, 16, 16, 8)); // Right eye region
        Mat forehead_region = gray_face(Rect(16, 4, 32, 12));  // Forehead region
        
        Scalar mean_mouth, stddev_mouth;
        meanStdDev(mouth_region, mean_mouth, stddev_mouth);
        
        Scalar mean_eye_left, stddev_eye_left;
        meanStdDev(left_eye_region, mean_eye_left, stddev_eye_left);
        
        Scalar mean_eye_right, stddev_eye_right;
        meanStdDev(right_eye_region, mean_eye_right, stddev_eye_right);
        
        Scalar mean_forehead, stddev_forehead;
        meanStdDev(forehead_region, mean_forehead, stddev_forehead);
        
        // Calculate face brightness
        Scalar mean_brightness, stddev_brightness;
        meanStdDev(gray_face, mean_brightness, stddev_brightness);
        
        // Measure face symmetry (difference between left and right sides)
        Mat left_half = gray_face(Rect(0, 0, 32, 64));
        Mat right_half = gray_face(Rect(32, 0, 32, 64));
        
        Scalar mean_left, stddev_left;
        meanStdDev(left_half, mean_left, stddev_left);
        
        Scalar mean_right, stddev_right;
        meanStdDev(right_half, mean_right, stddev_right);
        
        float face_symmetry = abs(mean_left[0] - mean_right[0]);
        
        // Calculate smile indicator (difference between upper and lower mouth regions)
        Mat upper_mouth = mouth_region(Rect(8, 0, 16, 8));
        Mat lower_mouth = mouth_region(Rect(8, 8, 16, 8));
        
        Scalar mean_upper, stddev_upper;
        meanStdDev(upper_mouth, mean_upper, stddev_upper);
        
        Scalar mean_lower, stddev_lower;
        meanStdDev(lower_mouth, mean_lower, stddev_lower);
        
        float smile_indicator = mean_lower[0] - mean_upper[0] + (stddev_lower[0] - stddev_upper[0]) * 2.0f;
        
        // Calculate advanced smile metric (gradient-based)
        Mat mouth_sobel_y;
        Sobel(mouth_region, mouth_sobel_y, CV_32F, 0, 1);
        Mat smile_curve;
        cv::reduce(mouth_sobel_y, smile_curve, 1, cv::REDUCE_AVG);
        float gradient_smile = 0.0f;
        if (smile_curve.rows > 2) {
            gradient_smile = smile_curve.at<float>(0) - smile_curve.at<float>(smile_curve.rows - 1);
        }
        
        // Calculate eye openness (for surprise, tired)
        float eye_left_openness = stddev_eye_left[0] / mean_eye_left[0];
        float eye_right_openness = stddev_eye_right[0] / mean_eye_right[0];
        float eye_openness = (eye_left_openness + eye_right_openness) / 2.0f;
        
        // Reset confidence scores for all emotions
        for (const auto& label : emotion_labels) {
            emotion_confidence[label] = 0.0f;
        }
        emotion_confidence["Tired"] = 0.0f;
        
        // Debug output
        if (debug_mode) {
            logger.debug("Smile indicator: " + to_string(smile_indicator));
            logger.debug("Gradient smile: " + to_string(gradient_smile));
            logger.debug("Mouth stddev: " + to_string(stddev_mouth[0]));
            logger.debug("Eye left stddev: " + to_string(stddev_eye_left[0]));
            logger.debug("Eye right stddev: " + to_string(stddev_eye_right[0]));
            logger.debug("Forehead stddev: " + to_string(stddev_forehead[0]));
            logger.debug("Face symmetry: " + to_string(face_symmetry));
            logger.debug("Mean brightness: " + to_string(mean_brightness[0]));
            logger.debug("Eye openness: " + to_string(eye_openness));
        }
        
        // Start with a moderate neutral baseline (not too high)
        emotion_confidence["Neutral"] = 0.75f;
        
        // Happy detection (based on smile indicator)
        // Lower the threshold for smile detection
        if (smile_indicator > 30.0f || gradient_smile > 10.0f) {
            float happy_conf = 0.5f;
            if (smile_indicator > 50.0f) happy_conf += 0.2f;
            if (gradient_smile > 20.0f) happy_conf += 0.1f;
            emotion_confidence["Happy"] = min(happy_conf, 0.9f);
            emotion_confidence["Neutral"] = max(0.2f, emotion_confidence["Neutral"] - 0.3f);
        }
        
        // Sad detection (negative smile indicator and higher forehead activity)
        if (smile_indicator < -15.0f || gradient_smile < -5.0f) {
            float sad_conf = 0.5f;
            if (smile_indicator < -25.0f) sad_conf += 0.15f;
            if (gradient_smile < -15.0f) sad_conf += 0.1f;
            emotion_confidence["Sad"] = min(sad_conf, 0.85f);
            emotion_confidence["Neutral"] = max(0.3f, emotion_confidence["Neutral"] - 0.25f);
        }
        
        // Angry detection (face asymmetry, negative smile, high forehead activity)
        if (face_symmetry > 25.0f && (smile_indicator < -10.0f || gradient_smile < -5.0f) && 
            stddev_forehead[0] > 45.0f) {
            float angry_conf = 0.45f;
            if (face_symmetry > 35.0f) angry_conf += 0.15f;
            if (stddev_forehead[0] > 55.0f) angry_conf += 0.15f;
            emotion_confidence["Angry"] = min(angry_conf, 0.85f);
            emotion_confidence["Neutral"] = max(0.25f, emotion_confidence["Neutral"] - 0.3f);
        }
        
        // Surprised detection (high eye and mouth stddev)
        if ((stddev_eye_left[0] > 60.0f && stddev_eye_right[0] > 60.0f) || 
            (eye_openness > 0.5f && stddev_mouth[0] > 65.0f)) {
            float surprised_conf = 0.55f;
            if (stddev_eye_left[0] > 65.0f && stddev_eye_right[0] > 65.0f) surprised_conf += 0.15f;
            if (stddev_mouth[0] > 70.0f) surprised_conf += 0.1f;
            emotion_confidence["Surprised"] = min(surprised_conf, 0.85f);
            emotion_confidence["Neutral"] = max(0.3f, emotion_confidence["Neutral"] - 0.3f);
        }
        
        // Tired detection (low eye stddev, overall low contrast)
        if ((stddev_eye_left[0] < 50.0f && stddev_eye_right[0] < 50.0f) &&
            mean_brightness[0] > 100.0f && stddev_brightness[0] < 60.0f) {
            float tired_conf = 0.5f;
            if (eye_openness < 0.4f) tired_conf += 0.15f;
            emotion_confidence["Tired"] = min(tired_conf, 0.8f);
            emotion_confidence["Neutral"] = max(0.35f, emotion_confidence["Neutral"] - 0.2f);
        }
        
        // Contempt (asymmetric face with specific mouth configuration)
        if (face_symmetry > 22.0f && smile_indicator < 0.0f && smile_indicator > -30.0f) {
            float contempt_conf = 0.4f;
            if (face_symmetry > 28.0f) contempt_conf += 0.1f;
            emotion_confidence["Contempt"] = min(contempt_conf, 0.75f);
            emotion_confidence["Neutral"] = max(0.4f, emotion_confidence["Neutral"] - 0.15f);
        }
        
        // Disgusted (mouth variations and asymmetry)
        if (stddev_mouth[0] > 62.0f && face_symmetry > 20.0f && smile_indicator < -5.0f) {
            float disgusted_conf = 0.45f;
            if (stddev_mouth[0] > 68.0f) disgusted_conf += 0.1f;
            emotion_confidence["Disgusted"] = min(disgusted_conf, 0.8f);
            emotion_confidence["Neutral"] = max(0.3f, emotion_confidence["Neutral"] - 0.25f);
        }
        
        // Fearful (wide eyes, low mouth variance)
        if (stddev_eye_left[0] > 55.0f && stddev_eye_right[0] > 55.0f && 
            stddev_mouth[0] < 60.0f && face_symmetry < 20.0f) {
            float fearful_conf = 0.4f;
            if (stddev_eye_left[0] > 65.0f && stddev_eye_right[0] > 65.0f) fearful_conf += 0.15f;
            emotion_confidence["Fearful"] = min(fearful_conf, 0.75f);
            emotion_confidence["Neutral"] = max(0.35f, emotion_confidence["Neutral"] - 0.2f);
        }
        
        // Debug output
        if (debug_mode) {
            string conf_str = "";
            for (const auto& conf : emotion_confidence) {
                conf_str += conf.first + ": " + to_string(conf.second) + " ";
            }
            logger.debug("Traditional confidence scores: " + conf_str);
        }
        
        // Find emotion with highest confidence
        string max_emotion = "Neutral";
        float max_conf = 0.0f;
        
        for (const auto& conf : emotion_confidence) {
            if (conf.second > max_conf) {
                max_conf = conf.second;
                max_emotion = conf.first;
            }
        }
        
        current_confidence = max_conf;
        
        if (debug_mode) {
            logger.debug("Detected mood: " + max_emotion + " with confidence: " + to_string(max_conf));
        }
        
        return max_emotion;
    }
    
    Scalar getEmotionColor(const string& emotion) {
        if (emotion_colors.find(emotion) != emotion_colors.end()) {
            return emotion_colors[emotion];
        }
        return Scalar(255, 255, 255); // Default to white
    }
    
    float getCurrentConfidence() {
        return current_confidence;
    }
    
    bool isUsingML() {
        return use_ml && model_loaded;
    }
};

// Initialize static member
bool MoodDetector::model_load_attempted = false;

// Parse command line arguments
void parseCommandLine(int argc, char** argv, Config& config, Logger& logger) {
    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        
        if (arg == "--camera" && i + 1 < argc) {
            config.camera_id = stoi(argv[++i]);
        }
        else if (arg == "--scale" && i + 1 < argc) {
            config.scale_factor = stod(argv[++i]);
        }
        else if (arg == "--debug") {
            config.debug_mode = true;
        }
        else if (arg == "--save") {
            config.save_detections = true;
        }
        else if (arg == "--dir" && i + 1 < argc) {
            config.save_directory = argv[++i];
        }
        else if (arg == "--record") {
            config.record_video = true;
        }
        else if (arg == "--output" && i + 1 < argc) {
            config.video_output = argv[++i];
        }
        else if (arg == "--confidence" && i + 1 < argc) {
            config.detection_confidence = stod(argv[++i]);
        }
        else if (arg == "--min-face-size" && i + 2 < argc) {
            int width = stoi(argv[++i]);
            int height = stoi(argv[++i]);
            config.min_face_size = Size(width, height);
        }
        else if (arg == "--process-all-frames") {
            config.process_every_frame = true;
        }
        else if (arg == "--emotion-model" && i + 1 < argc) {
            config.emotion_model_path = argv[++i];
        }
        else if (arg == "--no-ml") {
            config.use_ml_emotion = false;
        }
        else if (arg == "--help") {
            cout << "Face and Mood Detection - Help" << endl;
            cout << "---------------------------" << endl;
            cout << "Usage: ./face_mood_detection [options]" << endl;
            cout << "Options:" << endl;
            cout << "  --camera <id>          Camera device ID (default: 0)" << endl;
            cout << "  --scale <factor>       Scale factor for processing (default: 0.5)" << endl;
            cout << "  --debug                Enable debug mode" << endl;
            cout << "  --save                 Save detected faces" << endl;
            cout << "  --dir <path>           Directory to save detections (default: 'detections')" << endl;
            cout << "  --record               Record video output" << endl;
            cout << "  --output <filename>    Output video filename (default: 'output.mp4')" << endl;
            cout << "  --confidence <value>   Detection confidence (min neighbors, default: 4.0)" << endl;
            cout << "  --min-face-size <w> <h> Minimum face size to detect (default: 30 30)" << endl;
            cout << "  --process-all-frames   Process every frame (default: every other frame)" << endl;
            cout << "  --emotion-model <path> Path to emotion recognition model file (ONNX format)" << endl;
            cout << "  --no-ml                Disable ML-based emotion detection (use rule-based)" << endl;
            cout << "  --help                 Show this help message" << endl;
            exit(0);
        }
    }
}

// Main program
int main(int argc, char** argv) {
    // Create config with default values
    Config config;
    
    // Create a temporary logger for parsing command line args
    Logger logger("face_detection.log", false);
    
    // Parse command line arguments
    parseCommandLine(argc, argv, config, logger);
    
    // Create new logger with proper debug level 
    Logger logger_with_debug("face_detection.log", config.debug_mode);
    logger_with_debug.info("Starting Face and Mood Detection");
    
    // Show configuration
    logger_with_debug.info("Configuration:");
    logger_with_debug.info("  Camera ID: " + to_string(config.camera_id));
    logger_with_debug.info("  Scale factor: " + to_string(config.scale_factor));
    logger_with_debug.info("  Debug mode: " + string(config.debug_mode ? "enabled" : "disabled"));
    logger_with_debug.info("  Save detections: " + string(config.save_detections ? "enabled" : "disabled"));
    
    // Create save directory if needed
    if (config.save_detections) {
        try {
            if (!fs::exists(config.save_directory)) {
                fs::create_directories(config.save_directory);
                logger_with_debug.info("Created directory: " + config.save_directory);
            }
        }
        catch (const fs::filesystem_error& e) {
            logger_with_debug.error("Could not create directory: " + string(e.what()));
            config.save_detections = false;
        }
    }
    
    logger_with_debug.info("Attempting to initialize camera...");
    logger_with_debug.info("Note: On macOS, you may need to grant camera permission in System Preferences > Security & Privacy > Camera");
    
    // Initialize camera with preferred resolution
    VideoCapture cap(config.camera_id);
    if (!cap.isOpened()) {
        logger_with_debug.error("Could not open camera with ID: " + to_string(config.camera_id));
        logger_with_debug.error("If you're on macOS, please check that you've granted camera permissions:");
        logger_with_debug.error("1. Go to System Preferences > Security & Privacy > Camera");
        logger_with_debug.error("2. Ensure that Terminal (or your IDE) has permission to access the camera");
        logger_with_debug.error("3. You may need to restart your terminal or IDE after granting permission");
        return -1;
    }
    
    // Set optimal resolution - 640x480 is a good balance for performance
    cap.set(CAP_PROP_FRAME_WIDTH, 640);
    cap.set(CAP_PROP_FRAME_HEIGHT, 480);
    
    // Get actual camera resolution
    int cam_width = cap.get(CAP_PROP_FRAME_WIDTH);
    int cam_height = cap.get(CAP_PROP_FRAME_HEIGHT);
    logger_with_debug.info("Camera initialized with resolution: " + to_string(cam_width) + "x" + to_string(cam_height));
    
    // Initialize video writer if recording
    VideoWriter video_writer;
    if (config.record_video) {
        int codec = VideoWriter::fourcc('a', 'v', 'c', '1');  // H.264 codec
        double fps = 30.0;
        Size frame_size(cam_width, cam_height);
        
        video_writer.open(config.video_output, codec, fps, frame_size);
        
        if (!video_writer.isOpened()) {
            logger_with_debug.error("Could not open video writer. Recording disabled.");
            config.record_video = false;
        } else {
            logger_with_debug.info("Recording video to: " + config.video_output);
        }
    }
    
    // Load the cascade classifier for face detection
    CascadeClassifier face_cascade;
    string cascade_path = "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml";
    logger_with_debug.info("Loading face detection model from: " + cascade_path);
    
    if (!face_cascade.load(cascade_path)) {
        logger_with_debug.error("Could not load face cascade classifier.");
        logger_with_debug.error("Please check if the file exists at: " + cascade_path);
        return -1;
    }
    
    // Initialize mood detector
    MoodDetector mood_detector(logger_with_debug, config.debug_mode, config.emotion_model_path, config.use_ml_emotion);
    
    // Initialize variables outside the loop
    Mat frame, gray, small_frame;
    vector<Rect> faces;
    double scale_factor = config.scale_factor;
    double inv_scale = 1.0 / scale_factor;
    bool process_this_frame = true;
    
    // Performance metrics
    int frame_count = 0;
    int detection_count = 0;
    auto start_time = chrono::steady_clock::now();
    double fps = 0.0;
    
    // Create windows
    namedWindow("Face and Mood Detection", WINDOW_NORMAL);
    if (config.debug_mode) {
        namedWindow("Debug View", WINDOW_NORMAL);
    }
    
    logger_with_debug.info("Face and mood detection started. Press 'q' to quit, 's' to save current frame.");
    
    // Add instructions for calibration
    logger_with_debug.info("Calibration controls:");
    logger_with_debug.info("  '1' - Calibrate Neutral expression");
    logger_with_debug.info("  '2' - Calibrate Happy expression");
    logger_with_debug.info("  '3' - Calibrate Sad expression");
    logger_with_debug.info("  '4' - Calibrate Angry expression");
    logger_with_debug.info("  '5' - Calibrate Surprised expression"); 
    logger_with_debug.info("  '0' - Reset calibration");
    
    // Map keys to emotions for calibration
    map<int, string> key_to_emotion = {
        {'1', "Neutral"},
        {'2', "Happy"},
        {'3', "Sad"},
        {'4', "Angry"},
        {'5', "Surprised"},
        {'6', "Disgusted"},
        {'7', "Fearful"},
        {'8', "Contempt"},
        {'9', "Tired"}
    };
    
    while (true) {
        // Capture frame
        cap >> frame;
        if (frame.empty()) {
            logger_with_debug.error("Blank frame captured.");
            break;
        }
        
        // Record the frame if enabled
        if (config.record_video) {
            video_writer.write(frame);
        }
        
        // Update FPS calculation
        frame_count++;
        if (frame_count >= 10) {
            auto end_time = chrono::steady_clock::now();
            fps = frame_count * 1000.0 / chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count();
            frame_count = 0;
            start_time = end_time;
        }
        
        // Process this frame or skip based on configuration
        if (process_this_frame || config.process_every_frame) {
            // Resize frame for faster processing
            resize(frame, small_frame, Size(), scale_factor, scale_factor);
            
            // Convert to grayscale for face detection (faster)
            cvtColor(small_frame, gray, COLOR_BGR2GRAY);
            
            // Equalize histogram for better detection in varying lighting
            equalizeHist(gray, gray);
            
            // Clear previous detections
            faces.clear();
            
            // Detect faces with optimized parameters
            face_cascade.detectMultiScale(gray, faces, 
                                         1.1,      // Scale factor
                                         config.detection_confidence,  // Min neighbors (configurable)
                                         0,        // Flags
                                         config.min_face_size); // Min face size (configurable)
            
            detection_count += faces.size();
            
            if (config.debug_mode && !faces.empty()) {
                logger_with_debug.debug("Detected " + to_string(faces.size()) + " faces");
            }
        }
        process_this_frame = !process_this_frame;
        
        // Process each face
        for (const Rect& small_face : faces) {
            // Scale coordinates back to original size
            Rect face(cvRound(small_face.x * inv_scale),
                      cvRound(small_face.y * inv_scale),
                      cvRound(small_face.width * inv_scale),
                      cvRound(small_face.height * inv_scale));
            
            // Draw rectangle around the face
            rectangle(frame, face, Scalar(0, 255, 0), 2);
            
            // Extract face region for mood detection
            Rect safe_face = face & Rect(0, 0, frame.cols, frame.rows); // Ensure within image bounds
            if (safe_face.width > 0 && safe_face.height > 0) {
                Mat face_roi = frame(safe_face);
                
                // Create debug output if needed
                Mat debug_face;
                
                // Detect mood
                string mood = mood_detector.detectMood(face_roi, debug_face);
                Scalar mood_color = mood_detector.getEmotionColor(mood);
                float confidence = mood_detector.getCurrentConfidence();
                
                // Display the mood with appropriate color
                putText(frame, mood, Point(face.x, face.y - 10),
                        FONT_HERSHEY_SIMPLEX, 0.7, mood_color, 2);
                
                // Display confidence
                string conf_text = "Conf: " + to_string(int(confidence * 100)) + "%";
                putText(frame, conf_text, Point(face.x, face.y + face.height + 20),
                        FONT_HERSHEY_SIMPLEX, 0.5, mood_color, 1);
                
                // Show calibration status if active
                if (mood_detector.isUsingCalibration()) {
                    putText(frame, "Calibrated", Point(face.x, face.y - 30),
                            FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);
                }
                
                // Display debug view if enabled
                if (config.debug_mode && !debug_face.empty()) {
                    imshow("Debug View", debug_face);
                }
                
                // Save face detection if enabled
                if (config.save_detections) {
                    static int face_counter = 0;
                    string filename = config.save_directory + "/face_" + 
                                    to_string(face_counter++) + "_" + 
                                    mood + ".jpg";
                    try {
                        imwrite(filename, face_roi);
                        if (config.debug_mode) {
                            logger_with_debug.debug("Saved face to: " + filename);
                        }
                    }
                    catch (const cv::Exception& e) {
                        logger_with_debug.error("Failed to save face: " + string(e.what()));
                    }
                }
            }
        }
        
        // Display stats
        if (config.show_fps) {
            putText(frame, "FPS: " + to_string(int(fps)), Point(10, 30),
                    FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 255), 2);
            
            putText(frame, "Detections: " + to_string(detection_count), Point(10, 60),
                    FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 1);
        }
        
        // Display instructions
        putText(frame, "Press 'q' to quit, 's' to save frame, '1-9' to calibrate", Point(10, frame.rows - 10),
                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
        
        // Display calibration status
        if (mood_detector.isUsingCalibration()) {
            putText(frame, "Calibration: ACTIVE", Point(frame.cols - 200, 30),
                    FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 0), 2);
        }
        
        // Display output
        imshow("Face and Mood Detection", frame);
        
        // Handle keyboard input
        int key = waitKey(1);
        if (key == 'q' || key == 27) { // 'q' or ESC
            logger_with_debug.info("User requested exit");
            break;
        }
        else if (key == 's') { // 's' to save current frame
            string filename = "capture_" + to_string(chrono::system_clock::now().time_since_epoch().count()) + ".jpg";
            try {
                imwrite(filename, frame);
                logger_with_debug.info("Saved screenshot to: " + filename);
            }
            catch (const cv::Exception& e) {
                logger_with_debug.error("Failed to save screenshot: " + string(e.what()));
            }
        }
        else if (key == '0') { // Reset calibration
            mood_detector.resetCalibration();
            logger_with_debug.info("Emotion calibration reset");
        }
        else if (key_to_emotion.find(key) != key_to_emotion.end()) {
            // Calibrate for a specific emotion
            string emotion = key_to_emotion[key];
            
            // Make sure we have a face detected
            if (!faces.empty()) {
                Rect small_face = faces[0]; // Use the first face
                Rect face(cvRound(small_face.x * inv_scale),
                          cvRound(small_face.y * inv_scale),
                          cvRound(small_face.width * inv_scale),
                          cvRound(small_face.height * inv_scale));
                
                Rect safe_face = face & Rect(0, 0, frame.cols, frame.rows);
                if (safe_face.width > 0 && safe_face.height > 0) {
                    Mat face_roi = frame(safe_face);
                    
                    // Calibrate with this face for the selected emotion
                    if (mood_detector.calibrateEmotion(face_roi, emotion)) {
                        logger_with_debug.info("Calibrated for emotion: " + emotion);
                        
                        // Show calibration successful message
                        Mat calibration_display = frame.clone();
                        putText(calibration_display, "Calibrated for " + emotion, Point(frame.cols/2 - 150, frame.rows/2),
                                FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255, 0), 2);
                        imshow("Face and Mood Detection", calibration_display);
                        waitKey(1000); // Show message for 1 second
                    } else {
                        logger_with_debug.error("Failed to calibrate for emotion: " + emotion);
                    }
                }
            } else {
                logger_with_debug.error("No face detected for calibration");
            }
        }
    }
    
    // Release resources
    cap.release();
    if (config.record_video && video_writer.isOpened()) {
        video_writer.release();
        logger_with_debug.info("Video recording saved to: " + config.video_output);
    }
    destroyAllWindows();
    
    // Final statistics
    logger_with_debug.info("Session completed. Processed approximately " + 
               to_string(frame_count + detection_count) + " frames with " + 
               to_string(detection_count) + " face detections.");
    
    return 0;
} 