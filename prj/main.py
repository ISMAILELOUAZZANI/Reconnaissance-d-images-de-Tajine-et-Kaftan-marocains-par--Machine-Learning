# main.py
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.popup import Popup
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.widget import Widget
from kivy.graphics import Color, RoundedRectangle, Line
from kivy.core.window import Window
from kivy.clock import Clock
from kivy.animation import Animation
import tkinter as tk
from tkinter import filedialog

# Set window background color to a modern dark theme
Window.clearcolor = (0.08, 0.08, 0.12, 1)

# Load the trained model and label encoder
try:
    model = load_model('image_name_predictor.h5')
    label_encoder = np.load('label_encoder_classes.npy', allow_pickle=True)
    print("Model and label encoder loaded successfully")
except Exception as e:
    print(f"Error loading model or label encoder: {e}")
    model = None
    label_encoder = None

# Constants for colors and styling
DARK_BG = (0.08, 0.08, 0.12, 1)
PRIMARY_COLOR = (0.2, 0.6, 1, 1)
SUCCESS_COLOR = (0.2, 0.7, 0.3, 1)
ERROR_COLOR = (0.8, 0.3, 0.3, 1)
TEXT_COLOR = (0.8, 0.8, 0.9, 1)
PLACEHOLDER_COLOR = (0.5, 0.5, 0.6, 1)


def predict_image_name(img_path):
    """Predict the class of an image using the trained model."""
    if model is None:
        return "Error: Model not loaded"
    try:
        # Validate image path
        if not os.path.exists(img_path):
            return "Error: Image file does not exist"
            
        # Load and preprocess image
        img = cv2.imread(img_path)
        if img is None:
            return "Error: Could not load image"
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        
        # Make prediction
        prediction = model.predict(img)
        idx = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        predicted_class = label_encoder[idx]
        
        # Clean up
        del img
        return f"{predicted_class} ({confidence:.1f}% confidence)"
    except Exception as e:
        return f"Error during prediction: {str(e)}"

class RoundedButton(Button):
    """Custom rounded button with modern styling."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.background_color = (0, 0, 0, 0)
        self.bind(size=self.update_graphics, pos=self.update_graphics)
    def update_graphics(self, *args):
        self.canvas.before.clear()
        with self.canvas.before:
            Color(*self.bg_color if hasattr(self, 'bg_color') else (0.2, 0.6, 1, 1))
            RoundedRectangle(pos=self.pos, size=self.size, radius=[15])

class ImageDropArea(FloatLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bind(size=self.update_graphics, pos=self.update_graphics)
    def update_graphics(self, *args):
        self.canvas.before.clear()
        with self.canvas.before:
            Color(0.15, 0.15, 0.2, 1)
            RoundedRectangle(pos=self.pos, size=self.size, radius=[20])
            Color(0.3, 0.3, 0.4, 1)
            Line(rounded_rectangle=(self.x, self.y, self.width, self.height, 20), width=2)

class StatusLabel(Label):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.color = (0.8, 0.8, 0.9, 1)
        self.font_size = '16sp'
        self.bind(size=self.update_graphics, pos=self.update_graphics)
    def update_graphics(self, *args):
        self.canvas.before.clear()
        with self.canvas.before:
            Color(0.12, 0.12, 0.18, 0.8)
            RoundedRectangle(pos=(self.x - 10, self.y - 5), size=(self.width + 20, self.height + 10), radius=[10])

class ImageClassifierApp(App):
    def build(self):
        self.title = "AI Image Classifier"
        main_layout = BoxLayout(orientation='vertical', padding=20, spacing=15)
        # Header
        header = BoxLayout(orientation='vertical', size_hint=(1,0.15), spacing=5)
        header.add_widget(Label(text="🤖 AI Image Classifier", font_size='28sp', color=(1,1,1,1), bold=True))
        header.add_widget(Label(text="Upload an image to get AI-powered classification", font_size='14sp', color=(0.7,0.7,0.8,1)))
        main_layout.add_widget(header)
        # Image display
        self.image_container = ImageDropArea(size_hint=(1,0.5))
        self.image_display = Image(source='', size_hint=(0.9,0.9), pos_hint={'center_x':0.5,'center_y':0.5}, allow_stretch=True, keep_ratio=True)
        self.placeholder_label = Label(text="📁 No image selected\nClick 'Upload Image' to get started", font_size='16sp', color=(0.5,0.5,0.6,1), halign='center', valign='middle', size_hint=(1,1), pos_hint={'center_x':0.5,'center_y':0.5})
        self.placeholder_label.bind(size=self.placeholder_label.setter('text_size'))
        self.image_container.add_widget(self.placeholder_label)
        self.image_container.add_widget(self.image_display)
        main_layout.add_widget(self.image_container)
        # Buttons
        button_layout = BoxLayout(orientation='horizontal', size_hint=(1,0.12), spacing=15)
        self.upload_button = RoundedButton(text="📤 Upload Image", font_size='16sp', color=(1,1,1,1), bold=True)
        self.upload_button.bg_color=(0.2,0.7,0.3,1)
        self.upload_button.bind(on_press=self.upload_image)
        button_layout.add_widget(self.upload_button)
        self.predict_button = RoundedButton(text="🔍 Predict", font_size='16sp', color=(1,1,1,1), bold=True, disabled=True)
        self.predict_button.bg_color=(0.2,0.6,1,1)
        self.predict_button.bind(on_press=self.predict)
        button_layout.add_widget(self.predict_button)
        self.clear_button = RoundedButton(text="🗑️ Clear", font_size='16sp', color=(1,1,1,1), bold=True)
        self.clear_button.bg_color=(0.8,0.3,0.3,1)
        self.clear_button.bind(on_press=self.clear_image)
        button_layout.add_widget(self.clear_button)
        main_layout.add_widget(button_layout)
        # Status
        self.result_label = StatusLabel(text="🎯 Ready for prediction", size_hint=(1,0.15), halign='center', valign='middle')
        self.result_label.bind(size=self.result_label.setter('text_size'))
        main_layout.add_widget(self.result_label)
        main_layout.add_widget(Widget(size_hint=(1,0.08)))
        return main_layout

    def upload_image(self, instance):
        try:
            if os.name == 'nt':
                # Create tkinter root once and destroy it after use
                if not hasattr(self, '_tk_root'):
                    self._tk_root = tk.Tk()
                    self._tk_root.withdraw()
                
                file_path = filedialog.askopenfilename(
                    title="Select an Image",
                    filetypes=[
                        ("Image files", "*.png *.jpg *.jpeg *.bmp *.gif *.tiff"),
                        ("All files", "*.*")
                    ]
                )
                
                if hasattr(self, '_tk_root'):
                    self._tk_root.destroy()
                    del self._tk_root
                
                if file_path:
                    self.load_selected_image(file_path)
            else:
                self.show_file_chooser()
        except Exception as e:
            self.show_error(f"Error opening file dialog: {str(e)}")

    def show_file_chooser(self):
        chooser = FileChooserIconView(filters=['*.png','*.jpg','*.jpeg','*.bmp','*.gif','*.tiff'])
        popup = Popup(title="Select an Image", content=chooser, size_hint=(0.9,0.9))
        chooser.bind(on_submit=lambda inst, sel, touch: (self.load_selected_image(sel[0]) if sel else None, popup.dismiss()))
        popup.open()

    def load_selected_image(self, file_path):
        try:
            self.selected_image_path = file_path
            self.image_display.source = file_path; self.image_display.reload()
            self.placeholder_label.opacity = 0; self.image_display.opacity = 1
            self.predict_button.disabled = False; self.animate_button_enable(self.predict_button)
            self.result_label.text = f"📁 Image loaded: {os.path.basename(file_path)}\nReady for prediction!"
        except Exception as e:
            self.show_error(f"Error loading image: {e}")

    def clear_image(self, instance):
        # Clear image display
        self.image_display.source = ''
        self.image_display.opacity = 0
        self.placeholder_label.opacity = 1
        
        # Reset prediction button
        self.predict_button.disabled = True
        self.result_label.text = "🎯 Ready for prediction"
        
        # Clean up
        if hasattr(self, 'selected_image_path'):
            del self.selected_image_path
        
        # Force garbage collection
        import gc
        gc.collect()

    def predict(self, instance):
        """Performs prediction without closing the app."""
        try:
            if not hasattr(self, 'selected_image_path'):
                self.show_error("No image selected for prediction")
                return
            self.result_label.text = "🔄 Analyzing image..."; self.predict_button.disabled = True
            Clock.schedule_once(self.do_prediction, 0)
        except Exception as e:
            self.show_error(f"Prediction scheduling failed: {e}")

    def do_prediction(self, dt):
        try:
            result = predict_image_name(self.selected_image_path)
            self.result_label.text = f"🎯 Prediction: {result}"
            self.animate_result()
        except Exception as e:
            self.show_error(f"Prediction failed: {e}")
        finally:
            self.predict_button.disabled = False

    def show_error(self, message):
        self.result_label.text = f"❌ {message}"
        popup = Popup(title="Error", content=Label(text=message, text_size=(300,None), halign='center'), size_hint=(0.7,0.4))
        popup.open()

    def animate_button_enable(self, button):
        Animation(opacity=0.7, duration=0.2) + Animation(opacity=1, duration=0.2)

    def animate_result(self):
        Animation(font_size='18sp', duration=0.2) + Animation(font_size='16sp', duration=0.2)

if __name__ == '__main__':
    ImageClassifierApp().run()
