#let template(cfg: none, body) = {

    let title_page = {
        let top_banner = [

            #set par(
                justify: true,
                spacing: 0.65em,
            )
            #set text(size: 14pt)

            #cfg.university_name

            #cfg.faculty_name

            #cfg.edu_program_name
        ]

        let udk = align(left)[
            УДК #cfg.udk
        ]

        let theme_name = [
            #set text(weight: "bold")
            #cfg.project.name

            #set text(weight: "regular", size:12pt)
            (итоговый, этап 2)
        ]

        let student_info = align(left)[
            #set par(spacing: 1em)
            #set text(weight: "bold")

            Выполнил студент:
            
            #set text(weight: "regular", size:12pt)
            #table(
                columns: (auto, 1.5cm, auto),
                stroke: none,
                align: (left, center, left),
                [группы #cfg.student.group, #cfg.student.course курса], 
                [], 
                [#cfg.student.name]
            )
        ]

        let project_manager_info = align(left)[
            #set par(spacing: 1em)
            #set text(weight: "bold")

            Принял руководитель проекта:
            
            #set text(weight: "regular", 12pt)
            #table(
                columns: (auto),
                stroke: none,
                align: left,
                [#cfg.project_manager.name],
                [#cfg.project_manager.position]
            )

        ]

        let place_data = align(bottom)[
            #cfg.city #cfg.year
        ]

        page(
            header: none,
            footer: none,
            margin: (
                left: 25mm,
                right: 10mm,
                top: 20mm,
                bottom: 20mm,
            ),
        )[
            #set align(center)

            #grid(
                columns: 1fr,
                top_banner,
                v(20mm),
                udk,
                v(30mm),
                theme_name,
                v(25mm),
                student_info,
                v(10mm),
                project_manager_info,
                v(60mm),
                place_data,
            )
        ]

        counter(page).update(2)
    }


    

    
    let outline = {
        pagebreak(weak: true)

        {
            set align(center)
            set text(weight: "bold", size: 14pt)

            [Содержание]
        }

        set text(weight: "bold")
        outline(
            title: none,
            indent: 5mm,
            fill: none
        )
    }

    let outline_and_normal_pages = {
        set page(
            margin: (
                left: 25mm,
                right: 10mm,
                top: 20mm,
                bottom: 20mm,
            ),

            footer: [
                #set align(center)
                #context counter(page).display()
            ],
        )

        set par(
            justify: true,
            leading: 1em,
        )

        set heading(numbering: none)

        show heading.where(level: 1): h => {
            set align(center)
            set text(weight: "bold", size: 14pt)

            pagebreak(weak: true)
            h.body
            
        }

        show heading.where(level: 2): h => {
            set text(weight: "bold", size: 14pt)
            h.body
        }

        outline

        body
    }

    set text(
        lang: "ru",
        size: 14pt,
        font: "Times New Roman"
    )

    title_page
    outline_and_normal_pages
}
