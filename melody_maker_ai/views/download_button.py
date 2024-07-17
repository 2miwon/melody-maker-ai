import reflex as rx

def download_button(state: rx.State, result_fname: str) -> rx.Component:
    return rx.button(
            "Download Video with Music",
            # type="submit",
            border_radius="1em",
            box_shadow="rgba(151, 65, 252, 0.8) 0 15px 30px -10px",
            background_image="linear-gradient(144deg,#AF40FF,#5B42F3 50%,#00DDEB)",
            box_sizing="border-box",
            color="white",
            opacity=1,
            _hover={
                "opacity": 0.5,
            },
            size="3",
            disabled=~state.video_url,
            on_click=rx.download(url=state.video_url, filename=result_fname),
        )