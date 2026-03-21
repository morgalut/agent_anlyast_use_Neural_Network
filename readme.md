



```sh
uvicorn app.main:app --reload
```


```sh
curl -X POST "http://127.0.0.1:8000/run" -F "file=@multi_agent\data\Assembly - 1.xlsx"

  ```


# Let's start with a picture of the neural network itself.
  ![1774095371556](image/README/1774095371556.png)


----
# Now a diagram of the complete pipeline — what happens from the moment the file comes in until a response is received.
![1774095646213](image/README/1774095646213.png)


-----~~~
```sh
הסבר מלא — מה קורה בכל שלב
הרשת הנוירונית — הלוגיקה המרכזית
הרשת הנוירונית ב-PROMAT היא לא רשת למידה (אין משקלים שנלמדים). זוהי ארכיטקטורה לוגית ב-5 שכבות שמחקה את עקרונות הרשת הנוירונית — כל שכבה מעבדת את הפלט של הקודמת לה ומייצרת ייצוג מורכב יותר.
שכבה 0 — חילוץ קלט (OCR)
הסוכן פותח את קובץ האקסל עם data_only=False — חיוני כדי לקרוא את מחרוזות הנוסחאות עצמן ולא את התוצאות שלהן. מחלץ 8 תכונות גולמיות לכל גיליון: שם, האם מוסתר, כותרות, שורות לדוגמה, נוסחאות, גיליונות שהנוסחאות מפנות אליהם, ערכים שטוחים, והאם הגיליון הפעיל.
שכבה 1 — אקטיבציות בינאריות
9 אותות, כל אחד בדיוק 0 או 1. אין ציונים חלקיים. COA_SIGNAL דולק אם נמצאו לפחות 3 מ-7 הסעיפים הנדרשים. COMPANY_COLUMN_SIGNAL דולק אם כותרת עמודה מכילה NIS/$/ INC וכו'. CROSS_REF_SIGNAL דולק אם נוסחאות בגיליון מפנות לגיליון אחר. זה המנגנון שהחליף את מערכת הניקוד — במקום "20 נקודות ל-COA", כל אות הוא בינארי טהור.
שכבה 2 — לוגיקת דפוסים
שלושה דפוסים מחושבים מלוגיקה AND/OR/NOT של אותות שכבה 1. FS_PATTERN דורש שישה תנאים בו-זמנית: COA דולק, נוסחה קיימת, יש הפניה חוצה-גיליון, יש עמודת חברה, הגיליון לא מוסתר, ואין עמודת קוד. אם אפילו תנאי אחד כבוי — הדפוס כבוי. TB_PATTERN מזהה גיליון כרטסת. PARTIAL_FS_PATTERN מזהה ראיות חלקיות שדורשות אימות נוסף.
שכבה 3 — גרף הקשרי (cross-sheet)
כאן מתרחש הניתוח החכם ביותר. הסוכן בונה גרף מכוון בין כל הגיליונות: אם BS מכיל נוסחאות =SUMIF(SAP!I:I,...), אז יש קשת BS → SAP. לאחר מכן: BS.outgoing_refs = ["SAP"], SAP.incoming_refs = ["BS"]. מי שמפנה = גיליון FS. מי שמופנה אליו = גיליון TB. גרף זה מאפשר לזהות גם גיליון ביניים (FS → INTERMEDIATE → TB) ואת מצב ה-CONSOLIDATE.
שכבה 4 — שערים קשיחים
4 חומות אש לוגיות. אם גיליון מפעיל שער — הוא נפסל לחלוטין, ללא אפשרות פיצוי. GATE_2 הוא הקריטי ביותר: גיליון ללא עמודות חברה שאינו CONSOLIDATE — נחסם. זה מה שמנע את שגיאת ה-CF בדוגמה שראינו.
שכבה 5 — Softmax confidence
רק הגיליונות שעברו את שכבה 4 מקבלים ציון חוזק: S = FS_PATTERN×1.0 + PARTIAL×0.5 + AJE×0.2 + .... לאחר מכן normalisation: confidence = S / ΣS. התוצאה היא התפלגות הסתברות — לא ציון מוחלט. גיליון שהוא המועמד היחיד מקבל confidence=1.0 אוטומטית.

ה-Pipeline — מה קורה בכל node
analyze_node — פותח את הקובץ, מריץ inspect_workbook (שסורק את כל הגיליונות פיזית) ו-detect_main_sheet (גלאי היוריסטי). תוצאת הגלאי נשמרת כ-detector_candidate — רמז בלבד, לא מחויבות. main_sheet_name נשאר None. ה-LLM מריץ את שכבות ה-NN על הסיכום ומחזיר ניתוח JSON.
plan_node — מתרגם את פלט הניתוח לתוכנית אימות ממוקדת: אילו אותות לא נפתרו? איזה גיליון ביניים לבדוק? שולח הוראות ספציפיות לסוכן המחקר.
act_node — מריץ את research_agent שפותח את הקובץ ישירות עם כלי אקסל, מריץ את 5 שכבות ה-NN בעצמו, ומחזיר nn_evidence מלא עם כל האותות.
court_node — שלושה LLM שמתווכחים על פלט הסוכן. התובע (temperature=0.1) מחפש הפרות שכבות NN. הסנגור (temperature=0.7) מגן, אבל אסור לו להמציא עובדות. השופט (temperature=0.4) פוסק. אם הפסיקה היא revise_and_retry — הסוכן מתקן ומגיש מחדש, עד 2 ניסיונות.
synthesize_node — הבעלים של ההחלטה הסופית. מאגד nn_evidence מכל הסוכנים, מריץ softmax aggregation, ואז מפעיל את חומות האש ב-Python (_apply_nn_guardrails) — אלו בדיקות דטרמיניסטיות שאינן תלויות ב-LLM. כותב main_sheet_name ו-has_main_sheet לstate — שאר ה-nodes קוראים משם.
export_node — כותב קובץ JSON + Markdown עם כל ראיות שכבות ה-NN, לוג של הוויכוחים בבית המשפט, ושרשרת ההחלטות.

סיכום ה-PROMAT — כל הקבצים
nn_promat_core.py — מקור האמת. מכיל את 5 הגדרות השכבות וה-decision protocol. כל 7 קבצי ה-PROMAT מייבאים FULL_NN_PROMAT משם ומזריקים אותו כחלק עליון של ה-prompt.
analyze_prompt.py — מפעיל NN על סיכום הקובץ, מחזיר layer1_signals עד layer5_confidence לכל גיליון.
plan_prompt.py — הופך אותות לא-פתורים לצ'קליסט אימות.
task_prompt.py + research_prompt.py — מריצים NN ישירות על הקובץ עם כלי אקסל.
synthesize_prompt.py — מאגד NN מכל הסוכנים, softmax aggregation, החלטה סופית.
coder_prompt.py — ReAct עם קוד Python מוכן (extract_signals, compute_patterns, apply_gates, softmax_confidence).
critic_prompt.py — בודק עקביות שכבות NN. NC3 הוא הקריטי — הוא בודק שכל גיליון ללא COMPANY_COLUMN_SIGNAL נחסם ב-GATE_2, בדיוק מה שמנע את שגיאת CF.
```