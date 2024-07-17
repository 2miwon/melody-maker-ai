import reflex as rx

def pie_chart(data: list) -> rx.Component:
    return rx.recharts.pie_chart(
            rx.recharts.pie(
                data=data,
                data_key="value",
                name_key="name",
                cx="50%",
                cy="50%",
                # padding_angle=1,
                inner_radius="30",
                outer_radius="70",
                label=False,
            ),
            rx.recharts.graphing_tooltip(),
            rx.recharts.legend(),
            height=200,
        )