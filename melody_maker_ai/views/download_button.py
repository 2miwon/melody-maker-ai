import reflex as rx

def download_button(state: rx.State) -> rx.Component:
    return rx.button(
            "Download Video with Music",
            # type="submit",
            border_radius="1em",
            box_shadow="rgba(151, 65, 252, 0.8) 0 15px 30px -10px",
            background_image="linear-gradient(144deg,#AF40FF,#5B42F3 50%,#00DDEB)",
            box_sizing="border-box",
            color="white",
            opacity=rx.cond(state.output_video, 1.0, 0.5),
            # _hover={
            #     "opacity": 1.0
            # },
            size="3",
            disabled=~state.output_video,
            on_click=rx.download(url=state.output_video, filename="result.mp4"),
        )