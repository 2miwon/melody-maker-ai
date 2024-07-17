import reflex as rx

def upload_box(state: rx.State) -> rx.Component:
    return rx.upload(
        rx.text(
            "Drag and drop video or click to select video file\n max size: 10mb",
        ),
        accept={"video": ["video/*"]},
        id="my_upload",
        border="1px dotted rgb(107,99,246)",
        padding="5em",
        multiple=False,
        # on_drop=State.handle_drop(),
        on_drop=state.handle_upload(rx.upload_files(upload_id="my_upload")),
        disabled=False,
        max_size=10000000, # 10mb
    )