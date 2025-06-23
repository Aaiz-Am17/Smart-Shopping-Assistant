# gui_app.py
import customtkinter as ctk
from ac_model import ACModel
from tv_model import TVModel
from google.generativeai import Client as GenerativeAIClient

class ProductChoiceInterface:
    def __init__(self, root, ac_model_instance, tv_model_instance, ai_client):
        self.root = root
        self.ac_model = ac_model_instance
        self.tv_model = tv_model_instance
        self.ai_client = ai_client # Google Generative AI Client
        self.root.title("Smart Shopping Assistant")
        self.root.geometry("800x600")
        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")
        self.create_main_screen()

    def clear_screen(self):
        for widget in self.root.winfo_children():
            widget.destroy()

    def create_main_screen(self):
        self.clear_screen()
        main_frame = ctk.CTkFrame(self.root)
        main_frame.pack(fill="both", expand=True)

        ctk.CTkLabel(main_frame, text="Welcome to the Smart Shopping Assistant", font=ctk.CTkFont(size=20, weight="bold")).pack(pady=20)
        ctk.CTkLabel(main_frame, text="What would you like to buy?", font=ctk.CTkFont(size=16)).pack(pady=10)

        TV_button = ctk.CTkButton(main_frame, text="Smart TV", font=ctk.CTkFont(size=14), command=self.run_tv_assistant)
        TV_button.pack(pady=20)

        AC_button = ctk.CTkButton(main_frame, text="Air Conditioner", font=ctk.CTkFont(size=14), command=self.run_ac_assistant)
        AC_button.pack(pady=20)

    # --- TV Assistant Methods ---
    def run_tv_assistant(self):
        self.clear_screen()
        self.TV_current_step = 0
        self.TV_user_choices = {}
        self.TV_category_order = [
            ("TV_OS_Category", ['Android', 'Linux', 'Google TV', 'Other']),
            ("TV_Picture_Quality_Category", ['4K', 'Full HD', 'HD Ready', 'Other']),
            ("TV_Speaker_Output_Category", ['10-30W', '30-60W', '60-90W', '90+W']),
            ("Frequency", ['50Hz', '60Hz', '120Hz']),
            ("channel", ['Netflix', 'Prime Video', 'Disney+Hotstar', 'Youtube'])
        ]
        self.TV_create_loading_screen()

    def TV_create_loading_screen(self):
        TV_frame = ctk.CTkFrame(self.root)
        TV_frame.pack(fill="both", expand=True)
        ctk.CTkLabel(TV_frame, text="Smart TV Shopping Assistant", font=ctk.CTkFont(size=20, weight="bold")).pack(pady=20)
        ctk.CTkLabel(TV_frame, text=self.tv_model.load_message, font=ctk.CTkFont(size=12)).pack(pady=5)

        if self.tv_model.ensemble_model:
            ctk.CTkLabel(TV_frame, text="Performing RandomizedSearchCV for RandomForest...", font=ctk.CTkFont(size=12)).pack(pady=5)
            ctk.CTkLabel(TV_frame, text=f"Best Parameters for RandomForest: {self.tv_model.rf_best_params}", font=ctk.CTkFont(size=12)).pack(pady=5)
            ctk.CTkLabel(TV_frame, text=f"Mean R2 Score for RandomForest: {self.tv_model.rf_mean_r2:.3f}", font=ctk.CTkFont(size=12)).pack(pady=5)
            ctk.CTkLabel(TV_frame, text="Performing RandomizedSearchCV for XGBoost...", font=ctk.CTkFont(size=12)).pack(pady=5)
            ctk.CTkLabel(TV_frame, text=f"Best Parameters for XGBoost: {self.tv_model.xgb_best_params}", font=ctk.CTkFont(size=12)).pack(pady=5)
            ctk.CTkLabel(TV_frame, text=f"Mean R2 Score for XGBoost: {self.tv_model.xgb_mean_r2:.3f}", font=ctk.CTkFont(size=12)).pack(pady=5)
            ctk.CTkLabel(TV_frame, text="Ensemble Model is Ready!", font=ctk.CTkFont(size=14, weight="bold"), text_color="green").pack(pady=20)
            TV_start_button = ctk.CTkButton(TV_frame, text="Start Prediction",
                                            command=lambda: self.TV_show_next_category(TV_frame))
            TV_start_button.pack(pady=10)
        else:
            ctk.CTkLabel(TV_frame, text="TV Model not loaded/trained. Check dataset path or errors.",
                         font=ctk.CTkFont(size=14, weight="bold"), text_color="red").pack(pady=20)
            ctk.CTkButton(TV_frame, text="Back to Main Menu", command=self.create_main_screen).pack(pady=10)


    def TV_show_next_category(self, TV_frame):
        TV_frame.destroy()
        if self.TV_current_step < len(self.TV_category_order):
            TV_category, TV_options = self.TV_category_order[self.TV_current_step]
            self.TV_current_step += 1
            self.TV_create_category_screen(TV_category, TV_options)
        else:
            self.TV_predict_price()

    def TV_create_category_screen(self, TV_category, TV_options):
        TV_frame = ctk.CTkFrame(self.root)
        TV_frame.pack(fill="both", expand=True)
        ctk.CTkLabel(TV_frame, text=f"Select {TV_category}:", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=20)
        TV_selected_option = ctk.StringVar()
        for TV_option in TV_options:
            ctk.CTkRadioButton(TV_frame, text=TV_option, variable=TV_selected_option, value=TV_option).pack(anchor="w", padx=40)
        TV_next_button = ctk.CTkButton(TV_frame, text="Next",
                                   command=lambda: self.TV_save_choice_and_next(TV_frame, TV_category,
                                                                                TV_selected_option))
        TV_next_button.pack(pady=20)

    def TV_save_choice_and_next(self, TV_frame, TV_category, TV_selected_option):
        choice = TV_selected_option.get()
        if not choice:
            ctk.CTkMessagebox.showerror("Error", "Please select an option.")
            return
        self.TV_user_choices[TV_category] = choice
        self.TV_show_next_category(TV_frame)

    def TV_predict_price(self):
        TV_frame = ctk.CTkFrame(self.root)
        TV_frame.pack(fill="both", expand=True)

        try:
            TV_price = self.tv_model.predict_price(self.TV_user_choices)

            ctk.CTkLabel(TV_frame, text="Prediction Complete!", font=ctk.CTkFont(size=18, weight="bold"), text_color="green").pack(pady=20)
            ctk.CTkLabel(TV_frame, text=f"Predicted Price: Rs {TV_price:.2f}", font=ctk.CTkFont(size=24, weight="bold")).pack(pady=20)

            user_choices_summary = "\n".join([f"{k}: {v}" for k, v in self.TV_user_choices.items()])
            prompt = (
                f"I am looking for the best TV with the following qualities:\n{user_choices_summary}.\n"
                "Suggest the most suitable TV. Give me a single choice. "
                "If you can't suggest without screen size just say price may vary according to screen size but give me a single choice. "
                "Please keep the response concise."
            )

            try:
                response = self.ai_client.models.generate_content(
                    model="gemini-2.5-flash", contents=prompt
                )
                tv_recommendation = response.text
            except Exception as e:
                tv_recommendation = f"An error occurred while generating recommendations: {e}"

            ctk.CTkLabel(TV_frame, text="Recommendation Complete!", font=ctk.CTkFont(size=18, weight="bold"), text_color="green").pack(pady=20)
            ctk.CTkLabel(TV_frame, text="Recommended TV:", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)
            ctk.CTkLabel(TV_frame, text=tv_recommendation, font=ctk.CTkFont(size=14), wraplength=400).pack(pady=10)

        except ValueError as e:
            ctk.CTkLabel(TV_frame, text=f"Error predicting TV price: {e}", font=ctk.CTkFont(size=14, weight="bold"), text_color="red").pack(pady=20)
        except Exception as e:
            ctk.CTkLabel(TV_frame, text=f"An unexpected error occurred: {e}", font=ctk.CTkFont(size=14, weight="bold"), text_color="red").pack(pady=20)

        TV_restart_button = ctk.CTkButton(TV_frame, text="Restart", command=self.create_main_screen)
        TV_restart_button.pack(pady=20)

    # --- AC Assistant Methods ---
    def run_ac_assistant(self):
        self.clear_screen()
        self.AC_user_choices = {}
        self.AC_create_loading_screen()

    def AC_create_loading_screen(self):
        AC_frame = ctk.CTkFrame(self.root)
        AC_frame.pack(fill="both", expand=True)
        ctk.CTkLabel(AC_frame, text="Smart Air Conditioner Assistant", font=ctk.CTkFont(size=20, weight="bold")).pack(pady=20)
        ctk.CTkLabel(AC_frame, text=self.ac_model.load_message, font=ctk.CTkFont(size=12)).pack(pady=10)

        if self.ac_model.ensemble_model:
            ctk.CTkLabel(AC_frame, text=f"Best RandomForest R2: {self.ac_model.rf_mean_r2:.3f}", font=ctk.CTkFont(size=12)).pack(pady=5)
            ctk.CTkLabel(AC_frame, text=f"Best XGBoost R2: {self.ac_model.xgb_mean_r2:.3f}", font=ctk.CTkFont(size=12)).pack(pady=5)
            ctk.CTkLabel(AC_frame, text=f"Ensemble Model R2: {self.ac_model.ensemble_r2:.3f}", font=ctk.CTkFont(size=12)).pack(pady=5)
            AC_start_button = ctk.CTkButton(AC_frame, text="Start Prediction", command=self.AC_remove_loading_screen)
            AC_start_button.pack(pady=20)
        else:
            ctk.CTkLabel(AC_frame, text="AC Model not loaded/trained. Check dataset path or errors.",
                         font=ctk.CTkFont(size=14, weight="bold"), text_color="red").pack(pady=20)
            ctk.CTkButton(AC_frame, text="Back to Main Menu", command=self.create_main_screen).pack(pady=10)

    def AC_remove_loading_screen(self):
        self.clear_screen()
        self.AC_create_category_screen()

    def AC_create_category_screen(self):
        AC_frame = ctk.CTkFrame(self.root)
        AC_frame.pack(fill="both", expand=True)

        AC_categories = {
            "Condenser_Coil": self.ac_model.df['Condenser_Coil'].unique(),
            "Refrigerant": self.ac_model.df['Refrigerant'].unique(),
            "Power_Consumption": ['Low', 'Medium', 'High'],
            "Noise_level": ['Low', 'Medium', 'High']
        }

        # Storing selected option for each category using StringVar/OptionMenu
        self.ac_option_vars = {}

        for AC_category, AC_options in AC_categories.items():
            ctk.CTkLabel(AC_frame, text=f"Select {AC_category}:", font=ctk.CTkFont(size=14)).pack(pady=10)
            
            # Using CTkOptionMenu for dropdown selection
            var = ctk.StringVar(value=AC_options[0] if AC_options.size > 0 else "") # initial value
            option_menu = ctk.CTkOptionMenu(AC_frame, values=list(AC_options), variable=var)
            option_menu.pack(pady=5)
            self.ac_option_vars[AC_category] = var

        AC_submit_button = ctk.CTkButton(AC_frame, text="Predict Price", command=self.AC_finalize_input)
        AC_submit_button.pack(pady=20)

    def AC_finalize_input(self):
        required_categories = ['Condenser_Coil', 'Refrigerant', 'Power_Consumption', 'Noise_level']
        for cat in required_categories:
            self.AC_user_choices[cat] = self.ac_option_vars[cat].get()
            if not self.AC_user_choices[cat]:
                ctk.CTkMessagebox.showerror("Missing Information", f"Please select an option for {cat}.")
                return

        self.clear_screen()
        AC_result_frame = ctk.CTkFrame(self.root)
        AC_result_frame.pack(fill="both", expand=True)

        try:
            AC_predicted_price = self.ac_model.predict_price(self.AC_user_choices)
            AC_result_label = ctk.CTkLabel(AC_result_frame, text=f"Predicted Price: {AC_predicted_price:.2f}",
                                       font=ctk.CTkFont(size=24, weight="bold"))
            AC_result_label.pack(pady=20)

            user_choices_summary = "\n".join([f"{k}: {v}" for k, v in self.AC_user_choices.items()])
            prompt = (
                f"I am looking for the best 1.5 ton Air Conditioner with the following qualities:\n{user_choices_summary}.\n"
                "Suggest the most suitable 1.5 ton Air Conditioner to be used in a normal climate environment. "
                "Meaning, tell me which AC of which company should I go for. Do not give a detailed response."
            )

            try:
                response = self.ai_client.models.generate_content(
                    model="gemini-2.5-flash", contents=prompt
                )
                ac_recommendation = response.text
            except Exception as e:
                ac_recommendation = f"An error occurred while generating recommendations: {e}"

            ctk.CTkLabel(AC_result_frame, text="Recommendation Complete!", font=ctk.CTkFont(size=18, weight="bold"),
                         text_color="green").pack(pady=20)
            ctk.CTkLabel(AC_result_frame, text="Recommended AC:", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)
            ctk.CTkLabel(AC_result_frame, text=ac_recommendation, font=ctk.CTkFont(size=14), wraplength=400).pack(
                pady=10)

        except ValueError as e:
            ctk.CTkLabel(AC_result_frame, text=f"Error predicting AC price: {e}", font=ctk.CTkFont(size=14, weight="bold"), text_color="red").pack(pady=20)
        except Exception as e:
            ctk.CTkLabel(AC_result_frame, text=f"An unexpected error occurred: {e}", font=ctk.CTkFont(size=14, weight="bold"), text_color="red").pack(pady=20)

        AC_restart_button = ctk.CTkButton(AC_result_frame, text="Restart", command=self.create_main_screen)
        AC_restart_button.pack(pady=20)
