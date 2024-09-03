import gradio as gr


if __name__ == "__main__":
    def same_auth(username, password):
        if username not in ['admin', 'test']:
            return False
        return password == '123'

    with gr.Blocks() as demo:
        with gr.Tab("Pythera AI"):
            with gr.Row():
                with gr.Column(scale=1.5):
                    inp_typing = gr.Textbox(label="Typing content", lines=3)
                    with gr.Row():
                        upload_btn = gr.UploadButton(label='Upload File')
                        process_btn = gr.Button('ADD to DB')
                    threshold = gr.Slider(0.1, 1.0, value=0.87, label="Threshold", info="0.1 to 1.0", interactive=True),
                    topK = gr.Slider(1, 10, value=5, label="TopKNN", info="1 to 30", interactive= True),
                    clear = gr.ClearButton(value='CLEAR')
                    refresh = gr.ClearButton(value='REFRESH')
                with gr.Column(scale=3.5):
                    output = gr.Textbox(label="Result", lines=15)
                    with gr.Row():
                        inputs = gr.Textbox(label="Query")
                        text_button = gr.Button("Enter", size='sm', scale=0.1)
                    
    demo.queue(max_size=15)
    demo.launch(auth=same_auth, server_name="0.0.0.0", server_port=7860)