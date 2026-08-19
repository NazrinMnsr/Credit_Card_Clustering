<div align="center">

# Credit Card Customer Segmentation

**Unsupervised learning ilə müştəri seqmentasiyası və anomaliya aşkarlanması**

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9.0-F7931E?logo=scikitlearn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-success)

</div>

---

## İçindəkilər

- [Problem və Məqsəd](#-problem-və-məqsəd)
- [Dataset](#-dataset)
- [İş Axını](#-iş-axını)
- [Model Müqayisəsi](#-model-müqayisəsi)
- [Nəticələr: Müştəri Seqmentləri](#-nəticələr-müştəri-seqmentləri)
- [Əsas Tapıntılar](#-əsas-tapıntılar)
- [Quraşdırma və İstifadə](#-quraşdırma-və-istifadə)
- [Layihə Strukturu](#-layihə-strukturu)
- [Məhdudiyyətlər və Gələcək İşlər](#-məhdudiyyətlər-və-gələcək-işlər)

---

## Problem və Məqsəd

Kredit kartı şirkətləri üçün bütün müştərilərə eyni marketinq və risk strategiyası tətbiq etmək effektiv deyil. Bu layihə 8,950 müştərinin alış-veriş, ödəniş və nağd avans davranışına əsaslanaraq **fərqli, mənalı seqmentlər** müəyyən edir və hər seqment üçün **konkret biznes tövsiyəsi** hazırlayır (risk idarəetməsindən premium xidmətə qədər).

**Əsas suallar:**
- Müştərilər davranışlarına görə hansı təbii qruplara ayrılır?
- Hansı seqmentlər ən yüksək risk / ən yüksək dəyər daşıyır?
- Statistik anomaliyalar biznes baxımından nə deməkdir?

## Dataset

| | |
|---|---|
| **Mənbə** | [Kaggle – Credit Card Dataset for Clustering](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata) |
| **Ölçü** | 8,950 müştəri × 18 dəyişən |
| **Tip** | Kredit kartı istifadəçilərinin 6 aylıq davranış məlumatları |

> Dataset lisenziya səbəbindən repo-ya daxil edilməyib. Yuxarıdakı linkdən yükləyib `CC_GENERAL.csv` adı ilə layihə qovluğuna əlavə edin.

<details>
<summary><b>Dəyişənlərin tam siyahısı</b> (klikləyin)</summary>

| Dəyişən | Açıqlama |
|---|---|
| `BALANCE` | Cari kredit kartı balansı |
| `BALANCE_FREQUENCY` | Balansın yenilənmə tezliyi |
| `PURCHASES` | Ümumi alış-veriş məbləği |
| `ONEOFF_PURCHASES` | Birdəfəlik alışların məbləği |
| `INSTALLMENTS_PURCHASES` | Taksitli alışların məbləği |
| `CASH_ADVANCE` | Nağd avans məbləği |
| `PURCHASES_FREQUENCY` | Alış-veriş tezliyi |
| `CASH_ADVANCE_FREQUENCY` | Nağd avans tezliyi |
| `CASH_ADVANCE_TRX` | Nağd avans əməliyyat sayı |
| `PURCHASES_TRX` | Alış-veriş əməliyyat sayı |
| `CREDIT_LIMIT` | Kredit limiti |
| `PAYMENTS` | Edilmiş ümumi ödəniş |
| `MINIMUM_PAYMENTS` | Minimum ödəniş məbləği |
| `PRC_FULL_PAYMENT` | Balansı tam ödəmə faizi |
| `TENURE` | Kartdan istifadə müddəti (ay) |

</details>

## İş Axını

```
Data Cleaning → EDA (15 sual) → Feature Engineering → Outlier Analysis
      → Scaling (RobustScaler) → Clustering (4 metod) → Stability Testing
      → Anomaly Detection → Business Insights
```

| Mərhələ | Detallar |
|---|---|
| **Data Cleaning** | Boş dəyərlər median ilə dolduruldu (`CREDIT_LIMIT`, `MINIMUM_PAYMENTS`) |
| **EDA** | 15 biznes sualı üzərindən korrelyasiya və paylanma təhlili |
| **Feature Engineering** | `BALANCE_TO_LIMIT`, `TOTAL_TRANSACTIONS`, `HAS_CASH_ADVANCE`, `HAS_PURCHASES` (yalnız interpretasiya üçün — multicollinearity qarşısını almaq üçün clustering-dən çıxarıldı) |
| **Outlier Analysis** | IQR metodu; real davranış ehtimalı yüksək olduğundan silinmədi |
| **Scaling** | Outlier-lərə həssaslığı azaltmaq üçün `StandardScaler` əvəzinə `RobustScaler` |
| **Dimensionality** | PCA — **yalnız 2D/3D vizuallaşdırma üçün**, faktiki klasterləşmə tam ölçülü fəzada |
| **Validation** | Elbow, Silhouette (tam fəza + PCA fəzası), AIC/BIC, Hierarchical cross-tab, ARI stabillik testi |

## Model Müqayisəsi

| Metod | Silhouette Score | Nəticə |
|---|---:|---|
| **K-Means (k=5)** | 0.250 |  Seçildi — balanslı, interpretasiya edilə bilən klasterlər |
| Agglomerative (Ward) | 0.197 | Ward Hierarchical ilə güclü uyğunluq, doğrulama üçün istifadə edildi |
| Agglomerative (Single/Average) | 0.844 / 0.845 |  Süni yüksək skor — chaining effect, faydasız bölgü |
| DBSCAN | — |  eps dəyişdikcə qeyri-sabit klaster sayı |
| GMM (k=2–10) | ≤ 0.084 |  Aydın struktur tapılmadı; AIC/BIC k=9 tövsiyə etsə də praktik deyil |

## Nəticələr: Müştəri Seqmentləri

| # | Seqment | Ölçü | Xarakteristika | Tövsiyə |
|---|---|---:|---|---|
| 0 |  **Yüksək Risk** | 59 | Limitini keçib, minimum ödəniş ~22K, heç vaxt tam ödəməyib | Təcili risk monitorinqi, fərdi borc planı |
| 1 |  **Passiv İstifadəçi** | 1,608 | Aşağı balans və əməliyyat aktivliyi | Aktivləşdirmə kampaniyaları |
| 2 |  **Ultra-VIP** | 23 | Ən yüksək alış-veriş/ödəniş; 100% anomaliya | Fərdi relationship manager, premium kart |
| 3 |  **Aktiv/Yüksək Dəyərli** | 889 | Yüksək alış-veriş tezliyi | Cashback, taksit kampaniyaları |
| 4 |  **Standart Kütlə** | 6,371 (71%) | Orta-aşağı aktivlik, ən böyük qrup | Loyallıq proqramları, cross-sell |

## Əsas Tapıntılar

- **Dəyər = Anomaliya:** Ən dəyərli 23 müştəri (Cluster 2) eyni zamanda Isolation Forest tərəfindən **100% statistik anomaliya** kimi işarələnib — ekstremal davranış həm ən yaxşı, həm ən pis seqmentlərdə üzə çıxır.
- **Stabillik riski:** K-Means nəticələri `random_state`-dən asılıdır (10 seed üzrə orta ARI ≈ 0.33) — production üçün `n_init` artırılmalıdır.
- **Metod seçimi tək metrikaya əsaslanmayıb:** Elbow + Silhouette + Hierarchical cross-tab + AIC/BIC birgə istifadə olunaraq qərar gücləndirilib.

## Quraşdırma və İstifadə

```bash
git clone https://github.com/<username>/credit-card-segmentation.git
cd credit-card-segmentation
pip install -r requirements.txt
jupyter notebook Credit_Card_Dataset_for_Clustering.ipynb
```

Jupyter quraşdırmadan sürətli baxış üçün `Credit_Card_Dataset_for_Clustering.html` faylını birbaşa brauzerdə aça bilərsiniz — bütün kod, qrafiklər və nəticələr statik görünüşdə mövcuddur.

## Layihə Strukturu

```
credit-card-segmentation/
├── Credit_Card_Dataset_for_Clustering.ipynb   # Əsas analiz
├── Credit_Card_Dataset_for_Clustering.html    # Statik HTML görünüş
├── CC_GENERAL.csv                             # Kaggle-dan yüklənməlidir
├── requirements.txt
├── .gitignore
└── README.md
```

