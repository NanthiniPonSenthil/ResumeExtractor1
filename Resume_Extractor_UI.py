# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os
from Resume_Extractor import extract_skills_from_resume, extract_jd_with_genai, calculate_matching_percentage, check_experience_match

class ResumeExtractorUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Resume Extractor - Testing Interface")
        self.root.geometry("800x600")
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create tabs
        self.input_frame = ttk.Frame(self.notebook)
        self.results_frame = ttk.Frame(self.notebook)
        
        self.notebook.add(self.input_frame, text="Input")
        self.notebook.add(self.results_frame, text="Results")
        
        self.setup_input_tab()
        self.setup_results_tab()
        
        # Variables to store extraction results
        self.resume_data = None
        self.jd_data = None
        self.matching_results = None
        
    def setup_input_tab(self):
        # Resume file section
        resume_frame = ttk.LabelFrame(self.input_frame, text="Resume File", padding=10)
        resume_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.resume_path_var = tk.StringVar()
        ttk.Label(resume_frame, text="Select Resume File:").pack(anchor=tk.W)
        
        file_frame = ttk.Frame(resume_frame)
        file_frame.pack(fill=tk.X, pady=5)
        
        ttk.Entry(file_frame, textvariable=self.resume_path_var, width=50).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(file_frame, text="Browse", command=self.browse_resume_file).pack(side=tk.RIGHT, padx=(5, 0))
        
        # Job description section
        jd_frame = ttk.LabelFrame(self.input_frame, text="Job Description", padding=10)
        jd_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        ttk.Label(jd_frame, text="Paste Job Description:").pack(anchor=tk.W)
        
        self.jd_text = scrolledtext.ScrolledText(jd_frame, height=10, wrap=tk.WORD)
        self.jd_text.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Process button
        process_frame = ttk.Frame(self.input_frame)
        process_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.process_button = ttk.Button(process_frame, text="Extract Skills & Calculate Match", 
                                       command=self.process_documents, style="Accent.TButton")
        self.process_button.pack(side=tk.RIGHT)
        
        # Progress bar
        self.progress = ttk.Progressbar(process_frame, mode='indeterminate')
        self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
    def setup_results_tab(self):
        # Create scrollable frame for results
        canvas = tk.Canvas(self.results_frame)
        scrollbar = ttk.Scrollbar(self.results_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        self.results_container = scrollable_frame
        
        # Initial message
        self.no_results_label = ttk.Label(self.results_container, 
                                         text="No results yet. Process a resume and job description to see results here.",
                                         font=("Arial", 10))
        self.no_results_label.pack(pady=50)
        
    def browse_resume_file(self):
        file_types = [
            ("All Supported", "*.pdf;*.docx;*.doc;*.txt"),
            ("PDF files", "*.pdf"),
            ("Word files", "*.docx;*.doc"),
            ("Text files", "*.txt"),
            ("All files", "*.*")
        ]
        
        filename = filedialog.askopenfilename(
            title="Select Resume File",
            filetypes=file_types
        )
        
        if filename:
            self.resume_path_var.set(filename)
    
    def process_documents(self):
        # Validate inputs
        resume_path = self.resume_path_var.get().strip()
        jd_text = self.jd_text.get("1.0", tk.END).strip()
        
        if not resume_path:
            messagebox.showerror("Error", "Please select a resume file.")
            return
            
        if not os.path.exists(resume_path):
            messagebox.showerror("Error", "Resume file not found.")
            return
            
        if not jd_text:
            messagebox.showerror("Error", "Please enter a job description.")
            return
        
        # Start processing
        self.process_button.config(state="disabled")
        self.progress.start()
        
        try:
            # Extract from resume
            self.resume_data = extract_skills_from_resume(resume_path)
            
            # Extract from job description using GenAI
            jd_llm = extract_jd_with_genai(jd_text)
            skills = jd_llm.get("mandatory_skills", []) + jd_llm.get("non_mandatory_skills", [])
            self.jd_data = {
                "skills": skills,
                "years_experience": jd_llm.get("years_experience", ""),
                "mandatory_certifications": jd_llm.get("mandatory_certifications", []),
                "optional_certifications": jd_llm.get("non_mandatory_certifications", []),
                "mandatory_skills": jd_llm.get("mandatory_skills", []),
                "non_mandatory_skills": jd_llm.get("non_mandatory_skills", [])
            }
            
            # Calculate matching
            self.matching_results = calculate_matching_percentage(self.resume_data, self.jd_data)
            
            # Display results
            self.display_results()
            
            # Switch to results tab
            self.notebook.select(self.results_frame)
            
            messagebox.showinfo("Success", "Document processing completed successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during processing:\n{str(e)}")
        
        finally:
            self.progress.stop()
            self.process_button.config(state="normal")
    
    def display_results(self):
        # Clear existing results
        for widget in self.results_container.winfo_children():
            widget.destroy()
        
        if not self.resume_data or not self.matching_results:
            ttk.Label(self.results_container, text="No results to display.").pack(pady=20)
            return
        
        # Skills Extracted from Resume
        resume_skills_frame = ttk.LabelFrame(self.results_container, text="Skills Extracted from Resume", padding=10)
        resume_skills_frame.pack(fill=tk.X, padx=10, pady=5)
        # Show experience
        experience = self.resume_data.get("experience", "")
        ttk.Label(resume_skills_frame, text=f"Experience: {experience}", font=("Arial", 10)).pack(anchor=tk.W)

        # Mandatory / Non-mandatory skills
        resume_skills = self.resume_data.get("skills", [])
        if resume_skills:
            ttk.Label(resume_skills_frame, text=f"Mandatory/Detected Skills: {', '.join(resume_skills)}", font=("Arial", 10)).pack(anchor=tk.W)
        else:
            ttk.Label(resume_skills_frame, text="Mandatory/Detected Skills: None", font=("Arial", 10)).pack(anchor=tk.W)

        # Certifications
        certs = self.resume_data.get("certifications", [])
        if certs:
            ttk.Label(resume_skills_frame, text=f"Certifications: {', '.join(certs)}", font=("Arial", 10)).pack(anchor=tk.W)
        else:
            ttk.Label(resume_skills_frame, text="Certifications: None", font=("Arial", 10)).pack(anchor=tk.W)

        # Skills Extracted from JD
        jd_skills_frame = ttk.LabelFrame(self.results_container, text="Skills Extracted from JD", padding=10)
        jd_skills_frame.pack(fill=tk.X, padx=10, pady=5)

        # Experience
        jd_experience = self.jd_data.get("years_experience", "")
        ttk.Label(jd_skills_frame, text=f"Experience: {jd_experience}", font=("Arial", 10)).pack(anchor=tk.W)

        # Mandatory skills
        mandatory = self.jd_data.get("mandatory_skills", [])
        if mandatory:
            ttk.Label(jd_skills_frame, text=f"Mandatory Skills: {', '.join(mandatory)}", font=("Arial", 10)).pack(anchor=tk.W)
        else:
            ttk.Label(jd_skills_frame, text="Mandatory Skills: None", font=("Arial", 10)).pack(anchor=tk.W)

        # Non-mandatory skills
        non_mand = self.jd_data.get("non_mandatory_skills", [])
        if non_mand:
            ttk.Label(jd_skills_frame, text=f"Non-mandatory Skills: {', '.join(non_mand)}", font=("Arial", 10)).pack(anchor=tk.W)
        else:
            ttk.Label(jd_skills_frame, text="Non-mandatory Skills: None", font=("Arial", 10)).pack(anchor=tk.W)

        # Certifications
        jd_mand_certs = self.jd_data.get("mandatory_certifications", [])
        if jd_mand_certs:
            ttk.Label(jd_skills_frame, text=f"Mandatory Certifications: {', '.join(jd_mand_certs)}", font=("Arial", 10)).pack(anchor=tk.W)
        else:
            ttk.Label(jd_skills_frame, text="Mandatory Certifications: None", font=("Arial", 10)).pack(anchor=tk.W)

        jd_non_mand_certs = self.jd_data.get("optional_certifications", [])
        if jd_non_mand_certs:
            ttk.Label(jd_skills_frame, text=f"Non-mandatory Certifications: {', '.join(jd_non_mand_certs)}", font=("Arial", 10)).pack(anchor=tk.W)
        else:
            ttk.Label(jd_skills_frame, text="Non-mandatory Certifications: None", font=("Arial", 10)).pack(anchor=tk.W)

        # Experience Match Result
        exp_match_frame = ttk.LabelFrame(self.results_container, text="Experience Match Result", padding=10)
        exp_match_frame.pack(fill=tk.X, padx=10, pady=5)
        
        exp_matches = check_experience_match(self.resume_data, self.jd_data)
        if exp_matches:
            ttk.Label(exp_match_frame, text="Experience Matches", font=("Arial", 10, "bold"), foreground="green").pack(anchor=tk.W)
        else:
            ttk.Label(exp_match_frame, text="Doesn't match Experience", font=("Arial", 10, "bold"), foreground="red").pack(anchor=tk.W)

        # ...existing code...

def main():
    root = tk.Tk()
    app = ResumeExtractorUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()