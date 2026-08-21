# SMS Spam Detection – Naive Bayes (from scratch)

מטלה בלמידת מכונה — ניתוח טקסט (NLP), סיווג בינארי, Naive Bayes ממומש עצמאית (ללא `sklearn.naive_bayes`).

## מבנה הפרויקט

```
.
├── data/
│   └── spam.csv                         # SMS Spam Collection dataset (Kaggle)
├── notebooks/
│   ├── sms_spam_naive_bayes.ipynb       # המחברת המלאה, מורצת עם כל הפלטים
│   ├── sms_spam_naive_bayes.pdf         # תצוגה מודפסת של המחברת
│   └── spam.csv                          # עותק של הדאטה, כדי שהמחברת תרוץ עצמאית (למשל ב-Colab)
├── src/
│   ├── naive_bayes.py                    # המימוש הליבה: clean_and_tokenize + NaiveBayesTextClassifier
│   └── train.py                          # סקריפט אימון+הערכה עצמאי, ניתן להרצה מה-CLI
├── tests/
│   └── test_naive_bayes.py               # 19 בדיקות יחידה על המימוש
├── docs/
│   ├── מסמך_הסבר_על_הקוד.docx
│   └── תסריט_הסבר_לסרטון.md
├── requirements.txt
└── README.md
```

## איך להריץ ולוודא שהכל תקין

```bash
pip install -r requirements.txt

# בדיקות יחידה על מימוש האלגוריתם
pytest tests/ -v

# אימון + הערכה מלאים על הדאטהסט האמיתי
python src/train.py
```

## תוצאות (Test set, F1 על מחלקת ה-spam)

| מדד | ערך |
|---|---|
| F1 (spam) | 0.9273 |
| Precision (spam) | 0.9571 |
| Recall (spam) | 0.8993 |
| Accuracy | 0.9812 |
| Alpha שנבחר (Laplace smoothing) | 2.0 |

## אימות שבוצע (code scan)

- **19/19 בדיקות יחידה עוברות** — כיסוי ל: tokenization, priors נכונים, סיווג ברור נכון, טיפול במילים לא-מוכרות, Laplace smoothing שמונע הסתברות אפס, מניעת "דליפת מידע" (ה-vocabulary נבנה רק מה-train).
- **pyflakes** — 0 אזהרות (אין imports מיותרים, משתנים לא בשימוש, שגיאות תחביר).
- **הרצת אימון מלאה על הדאטה האמיתי** (`src/train.py`) עוברת בהצלחה מקצה לקצה, כולל אסרציות פנימיות (priors מסתכמים ל-1, ה-vocabulary לא ריק, F1 לא נופל מתחת לסף סביר).
- **המחברת (`notebooks/*.ipynb`) הורצה מחדש מאפס** ואומתה שאין אף תא עם שגיאה.
