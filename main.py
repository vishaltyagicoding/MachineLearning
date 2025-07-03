import webbrowser

url = "https://www.youtube.com/watch?v=2GV_ouHBw30&list=PLKnIA16_RmvbYFaaeLY28cWeqV-3vADST"  # Replace with the desired URL

# Open in the default browser in a new window if possible
webbrowser.open(url)

# Open in a new window, regardless of existing windows
webbrowser.open_new(url)

# Open in a new tab
webbrowser.open_new_tab(url)
