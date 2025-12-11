

---

## 📄 **VS Code Python Path Behavior — Essential Guide for Students**

### **1. Short Summary (Core Principle)**

VS Code loads environment settings (including `PYTHONPATH`) **only from the folder you open as the workspace**.
This means:

* When you open a folder in VS Code using **File → Open Folder…**, VS Code activates any `.vscode/settings.json` inside *that* folder.
* The PYTHONPATH defined there is applied **only to new terminals created after the folder is opened**.
* Changing directory (`cd`) inside an existing terminal does **not** load settings from subfolders.

To ensure your Python scripts can import your custom modules, always open the project folder (the one containing `.vscode/settings.json`) as your workspace **before** running code or opening terminals.

---

### **2. Key Points You Must Know**

* VS Code does **not** automatically load settings from subfolders when you `cd` into them.

* Terminals inherit environment variables (like `PYTHONPATH`) **only when they are created**, not afterward.

* To guarantee correct imports, always:
  
  * Open the correct folder as the workspace
  * Use a **new terminal**
  * **Do not rely on old terminals** after switching folders or changing settings

---

### **3. Quick Checklist (Follow in Order)**

1. **Open the correct project folder**
   
   * Use: **File → Open Folder… → (your project folder)**
   * This must be the folder that contains your `.vscode/settings.json`.

2. **Create a fresh terminal**
   
   * Close any existing terminals
   * Then open a new one: **Terminal → New Terminal**

3. **Verify PYTHONPATH**
   
   ```bash
   echo $PYTHONPATH
   ```
   
   * It should show the directories specified in your `.vscode/settings.json`.

4. **Run your Python file**
   
   * Use the green **Run** arrow **or** press **F5**.
   * If imports fail, return to steps 1–3.

5. **Important:**
   
   * If you run `unset PYTHONPATH` in the terminal, the editor Run button will no longer work for imports.
   * To fix: **close the terminal and open a new one**.

---

#### 👍 You now understand how VS Code decides when to apply your Python path settings.

This will save you hours of debugging “ModuleNotFoundError” issues and help keep your projects organized and working smoothly!

---


