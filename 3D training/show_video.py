"""
Helper to display or open a training video from the `3D training/videos/` folder.

Usage:
 - In a Jupyter notebook: from IPython.display import HTML, display; import show_video; display(show_video.show_video_html('videos/training_video.mp4'))
 - From terminal: python show_video.py videos/training_video.mp4

If running on Windows, the script will try to open the file with the default application.
"""
import os
import sys

def show_video_html(path):
    """Return an IPython HTML element string to embed an mp4 in a notebook."""
    try:
        from IPython.display import HTML
    except Exception:
        return "IPython not available. Use the script from a notebook to get embeds."

    if not os.path.exists(path):
        return HTML(f"<p>Video not found: {path}</p>")

    video_b64 = None
    try:
        import base64
        with open(path, 'rb') as f:
            data = f.read()
        data_url = "data:video/mp4;base64," + base64.b64encode(data).decode()
        html = f"<video width=640 controls><source src='{data_url}' type='video/mp4'></video>"
        return HTML(html)
    except Exception as e:
        return HTML(f"<p>Could not embed video: {e}</p>")

def open_with_default_app(path):
    if not os.path.exists(path):
        print(f"Video not found: {path}")
        return
    if sys.platform.startswith('win'):
        os.startfile(path)
    elif sys.platform.startswith('darwin'):
        os.system(f"open '{path}'")
    else:
        # linux
        os.system(f"xdg-open '{path}'")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python show_video.py <path/to/video.mp4>")
        sys.exit(1)
    video_path = sys.argv[1]
    open_with_default_app(video_path)
