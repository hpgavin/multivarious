---

# 📘 **VS Code + PYTHONPATH Cheat Sheet**

### *Understanding how VS Code decides when your Python modules are importable*

---

# 🧠 **Core Principle (The One Rule That Matters)**

> **VS Code only loads environment settings (like `PYTHONPATH`) from the folder you open as the workspace.**
> 
> If you don’t open that folder, VS Code will not apply the `.vscode/settings.json` inside it.

---

# 🔍 **How VS Code Sees the World**

### **1. When you open a folder as the VS Code workspace**

```
+-------------------------------+
| VS Code Workspace = /project |
+-------------------------------+

VS Code loads:
    /project/.vscode/settings.json
```

### **2. When you open a terminal AFTER opening the workspace**

```
Terminal #1 created AFTER opening workspace
-------------------------------------------
PYTHONPATH = (loaded from settings.json)
sys.path includes your module directories
```

### **3. If you cd into a subfolder**

```
cd project/subfolder
(no settings loaded here)
```

➡️ **VS Code does NOT re-check subfolder settings automatically.**

---

# 🔥 **Correct Behavior Diagram**

```
File → Open Folder →  /home/student/project
                ↓
VS Code loads project/.vscode/settings.json
                ↓
Terminal → New Terminal
                ↓
echo $PYTHONPATH  → shows correct value
                ↓
Run script → imports succeed
```

---

# ⚠️ **Incorrect Behavior Diagram**

```
File → Open Folder →  /home/student/          (WRONG)
                ↓
Terminal → New Terminal
                ↓
cd project
                ↓
echo $PYTHONPATH → still empty  (expected!)
                ↓
Run script → imports fail (nothing loaded)
```

Why?
Because **VS Code never read project/.vscode/settings.json** — you opened the wrong folder.

---

# 🏁 **Student Quick Checklist**

### ✔️ 1. OPEN the correct folder

Use: **File → Open Folder → (folder containing `.vscode/settings.json`)**

### ✔️ 2. OPEN a NEW terminal

Settings are applied only at terminal creation time.

### ✔️ 3. VERIFY the path

```
echo $PYTHONPATH
```

### ✔️ 4. RUN your script

Use:

* The **green run arrow**, or
* **F5** (recommended)

If imports fail → restart at Step 1.

---

# 🧪 **Testing Your Setup**

Put this at the top of your Python script:

```python
import os, sys
print("PYTHONPATH =", os.environ.get("PYTHONPATH"))
print("sys.path =", sys.path[:5])
```

If you don’t see your module directories:
➡️ You opened the wrong folder, or need a new terminal.

---

# 🛠️ **Typical Working Folder Layout**

```
Code/
│
├── modules/
│      ├── module_1/
│      └── module_2/
│
└── projects/
       └── project_A/
             ├── main.py
             └── .vscode/
                    └── settings.json
```

Students must open:

```
/home/student/Code/projects/project_A
```

NOT:

```
/home/student/Code
/home/student
/
```

---

# 💡 **Pro Tips**

### ⭐ Always close old terminals after changing folders

New settings are only applied to new terminals.

### ⭐ If all else fails, run your code using Run > Run without debugging  - function key [F5]

This uses the included `.vscode/launch.json` which injects PYTHONPATH directly into the debugger-run process.   --- be sure to first edit the <USERNAME> and path in `.vscode/launch.json` to match your login ID and your installation path.  

---

# 🎉 Done!
