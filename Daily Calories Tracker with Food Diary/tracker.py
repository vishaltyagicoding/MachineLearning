import tkinter as tk
from tkinter import messagebox, filedialog
from datetime import datetime
import json
import os

class CalorieTracker:
    def __init__(self, root):
        self.root = root
        self.root.title("Daily Calorie Tracker")
        self.root.geometry("500x600")
        self.root.configure(bg="#f0f8ff")
        
        # Current date
        self.current_date = datetime.now().strftime("%Y-%m-%d")
        
        # Data storage
        self.entries = []
        self.total_calories = 0
        
        self.setup_ui()
        self.load_from_file()
        
    def setup_ui(self):
        # Header
        header_frame = tk.Frame(self.root, bg="#2c3e50", height=80)
        header_frame.pack(fill="x", pady=(0, 10))
        header_frame.pack_propagate(False)
        
        title = tk.Label(header_frame, text="Daily Calorie Tracker", 
                        font=("Arial", 20, "bold"), fg="white", bg="#2c3e50")
        title.pack(expand=True)
        
        date_label = tk.Label(header_frame, text=f"Date: {self.current_date}", 
                             font=("Arial", 12), fg="white", bg="#2c3e50")
        date_label.pack(expand=True)
        
        # Input section
        input_frame = tk.Frame(self.root, bg="#f0f8ff", padx=20, pady=10)
        input_frame.pack(fill="x", pady=(0, 10))
        
        tk.Label(input_frame, text="Food Item:", font=("Arial", 12), 
                bg="#f0f8ff").grid(row=0, column=0, sticky="w", pady=5)
        self.food_entry = tk.Entry(input_frame, font=("Arial", 12), width=20)
        self.food_entry.grid(row=0, column=1, padx=5, pady=5)
        
        tk.Label(input_frame, text="Calories:", font=("Arial", 12), 
                bg="#f0f8ff").grid(row=1, column=0, sticky="w", pady=5)
        self.calorie_entry = tk.Entry(input_frame, font=("Arial", 12), width=10)
        self.calorie_entry.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        
        add_button = tk.Button(input_frame, text="Add Entry", font=("Arial", 12), 
                              bg="#3498db", fg="white", command=self.add_entry)
        add_button.grid(row=2, column=0, columnspan=2, pady=10)
        
        # Entries list
        list_frame = tk.Frame(self.root, bg="white", relief="sunken", bd=1)
        list_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Create a canvas and scrollbar for the list
        self.canvas = tk.Canvas(list_frame, bg="white")
        scrollbar = tk.Scrollbar(list_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas, bg="white")
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)
        
        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Total calories display
        total_frame = tk.Frame(self.root, bg="#f0f8ff", height=50)
        total_frame.pack(fill="x", pady=(0, 10))
        total_frame.pack_propagate(False)
        
        self.total_label = tk.Label(total_frame, text="Total Calories: 0", 
                                   font=("Arial", 14, "bold"), bg="#f0f8db", fg="#2c3e50")
        self.total_label.pack(expand=True)
        
        # Buttons frame
        button_frame = tk.Frame(self.root, bg="#f0f8ff", pady=10)
        button_frame.pack(fill="x")
        
        save_button = tk.Button(button_frame, text="Save to File", font=("Arial", 12), 
                               bg="#27ae60", fg="white", command=self.save_to_file)
        save_button.pack(side="left", expand=True, padx=10)
        
        clear_button = tk.Button(button_frame, text="Clear All", font=("Arial", 12), 
                                bg="#e74c3c", fg="white", command=self.clear_all)
        clear_button.pack(side="left", expand=True, padx=10)
        
        load_button = tk.Button(button_frame, text="Load from File", font=("Arial", 12), 
                               bg="#f39c12", fg="white", command=self.load_from_file_dialog)
        load_button.pack(side="left", expand=True, padx=10)
    
    def add_entry(self):
        food = self.food_entry.get().strip()
        calories_str = self.calorie_entry.get().strip()
        
        if not food or not calories_str:
            messagebox.showerror("Error", "Please enter both food item and calories.")
            return
        
        try:
            calories = int(calories_str)
            if calories <= 0:
                messagebox.showerror("Error", "Calories must be a positive number.")
                return
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number for calories.")
            return
        
        # Add to entries list
        entry = {"food": food, "calories": calories}
        self.entries.append(entry)
        self.total_calories += calories
        
        # Update UI
        self.update_entries_list()
        self.update_total_label()
        
        # Clear input fields
        self.food_entry.delete(0, tk.END)
        self.calorie_entry.delete(0, tk.END)
    
    def update_entries_list(self):
        # Clear the scrollable frame
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        
        # Add entries to the list
        for i, entry in enumerate(self.entries):
            entry_frame = tk.Frame(self.scrollable_frame, bg="white")
            entry_frame.pack(fill="x", padx=10, pady=5)
            
            tk.Label(entry_frame, text=entry["food"], font=("Arial", 12), 
                    bg="white", width=25, anchor="w").pack(side="left")
            
            tk.Label(entry_frame, text=f"{entry['calories']} cal", font=("Arial", 12), 
                    bg="white", width=10, anchor="w").pack(side="left")
            
            delete_btn = tk.Button(entry_frame, text="Delete", font=("Arial", 10), 
                                  bg="#ff6b6b", fg="white", 
                                  command=lambda idx=i: self.delete_entry(idx))
            delete_btn.pack(side="right", padx=5)
    
    def delete_entry(self, index):
        # Subtract calories from total
        self.total_calories -= self.entries[index]["calories"]
        
        # Remove entry
        del self.entries[index]
        
        # Update UI
        self.update_entries_list()
        self.update_total_label()
    
    def update_total_label(self):
        self.total_label.config(text=f"Total Calories: {self.total_calories}")
    
    def clear_all(self):
        if not self.entries:
            return
            
        if messagebox.askyesno("Confirm", "Are you sure you want to clear all entries?"):
            self.entries = []
            self.total_calories = 0
            self.update_entries_list()
            self.update_total_label()
    
    def save_to_file(self):
        if not self.entries:
            messagebox.showwarning("Warning", "No entries to save.")
            return
            
        # Create data structure
        data = {
            "date": self.current_date,
            "entries": self.entries,
            "total_calories": self.total_calories
        }
        
        # Suggest filename with current date
        default_filename = f"calories_{self.current_date}.json"
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialfile=default_filename
        )
        
        if filepath:
            try:
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=4)
                messagebox.showinfo("Success", f"Data saved to {filepath}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save file: {str(e)}")
    
    def load_from_file(self):
        # Try to load today's file automatically
        default_filename = f"calories_{self.current_date}.json"
        if os.path.exists(default_filename):
            try:
                with open(default_filename, 'r') as f:
                    data = json.load(f)
                
                self.entries = data["entries"]
                self.total_calories = data["total_calories"]
                
                self.update_entries_list()
                self.update_total_label()
            except:
                # If loading fails, just start fresh
                pass
    
    def load_from_file_dialog(self):
        filepath = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filepath:
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                self.entries = data["entries"]
                self.total_calories = data["total_calories"]
                
                self.update_entries_list()
                self.update_total_label()
                
                messagebox.showinfo("Success", f"Data loaded from {filepath}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = CalorieTracker(root)
    root.mainloop()