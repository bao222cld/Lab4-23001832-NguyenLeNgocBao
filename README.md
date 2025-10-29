# LAB 4 – Text Classification

## 1. Mục tiêu
Bài này nhằm xây dựng một hệ thống phân loại văn bản (text classification) sử dụng các phương pháp học máy cơ bản.  
Các bước thực hiện gồm:
- Tiền xử lý và vector hóa dữ liệu văn bản.
- Huấn luyện các mô hình phân loại với nhiều phương pháp khác nhau.
- Đánh giá và so sánh kết quả giữa các mô hình.
- Tối ưu hóa mô hình bằng Grid Search để cải thiện hiệu suất.

---

## 2. Dữ liệu
- Sử dụng dataset: **zeroshot/twitter-financial-news-sentiment** từ Hugging Face.  
- Đây là tập dữ liệu gồm các dòng tweet về tài chính, được gán nhãn cảm xúc (positive / negative / neutral).  
- Khi không tải được dataset gốc, chương trình có cơ chế **fallback** sử dụng một tập dữ liệu nhỏ minh họa để có thể chạy được toàn bộ pipeline.

---

## 3. Cấu trúc thư mục
Cấu trúc các file trong bài làm như sau:

├── text_classifier.py # Lớp TextClassifier, huấn luyện và đánh giá mô hình
├── vectorizers.py # Các bộ vector hóa: TF-IDF, Word2Vec
├── run_experiments.py # Script chạy toàn bộ thí nghiệm và lưu kết quả
├── lab5_test.py # Kiểm thử cơ bản pipeline
├── lab5_improvement_test.py # Kiểm thử mô hình cải tiến
├── lab5_spark_sentiment_analysis.py # Kiểm thử Spark (placeholder)
├── results.csv # Kết quả thực nghiệm (accuracy, precision, recall, f1)
├── results.json # Kết quả lưu dạng JSON
├── requirements.txt # Danh sách thư viện cần cài đặt
└── README.md # Báo cáo mô tả bài làm

---

## 4. Cách chạy
### Cài đặt môi trường
```bash
pip install -r requirements.txt
python run_experiments.py
Kết quả của các mô hình sẽ được lưu trong thư mục:
outputs/results.csv
outputs/results.json
5. Kết quả thực nghiệm (sample_size = 2000)
Mô hình	Accuracy	F1-score	Ghi chú
TF-IDF + Logistic Regression	0.7325	0.5799	Baseline
TF-IDF + MultinomialNB	0.6575	0.3357	Naive Bayes kém hơn
TF-IDF + Logistic (GridSearch, C=10)	0.7425	0.6190	Mô hình tốt nhất
Word2Vec-average + LogisticRegression	0.6450	0.2614	Embedding trung bình mất ngữ cảnh
6. Phân tích kết quả

GridSearch với Logistic Regression (C=10) đạt hiệu suất tốt nhất.
Việc điều chỉnh siêu tham số C giúp mô hình cân bằng giữa độ phức tạp và khả năng tổng quát, cải thiện đáng kể F1-score so với baseline.

Naive Bayes hoạt động kém hơn trong trường hợp này vì giả định độc lập giữa các đặc trưng không phù hợp với không gian TF-IDF.

Word2Vec trung bình cho kết quả thấp do phương pháp lấy trung bình embedding làm mất ngữ cảnh của từ trong câu.
Mô hình này sẽ cải thiện nếu dùng pre-trained embeddings (như Google News hoặc GloVe) hoặc mô hình ngữ cảnh như BERT.

7. Hướng cải thiện

Thêm bước preprocessing nâng cao: loại bỏ stopwords, stemming, lemmatization.

Dùng class balancing hoặc class_weight để xử lý mất cân bằng nhãn.

Thử nghiệm với mô hình Transformer (BERT) để cải thiện khả năng hiểu ngữ cảnh.

Kết hợp TF-IDF và Word2Vec làm đặc trưng lai (hybrid feature representation).

8. Kết luận

Bài làm đã xây dựng thành công pipeline phân loại văn bản hoàn chỉnh, có thể huấn luyện và đánh giá nhiều mô hình khác nhau.
Mô hình Logistic Regression được tối ưu bằng Grid Search cho kết quả tốt nhất với độ chính xác 74.25% và F1-score 0.619.
Kết quả phản ánh rõ sự khác biệt về hiệu năng giữa các mô hình và vai trò quan trọng của việc lựa chọn đặc trưng cùng tham số tối ưu.

9. Tài liệu tham khảo

Hugging Face Datasets: https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment

Thư viện: scikit-learn, gensim, datasets, joblib, pytest

Scikit-learn Documentation: https://scikit-learn.org/stable/
