import sympy
import matplotlib.pyplot as plt
import numpy as np
import gradio as gr

# Variable simbólica global
z = sympy.symbols('z')

# --- Función para Renderizar LaTeX como Imagen ---
def render_latex_formula(latex_str, fontsize=22):
    fig = plt.figure(figsize=(8, 2))
    ax = fig.add_subplot(111)
    ax.axis('off')
    ax.text(0.5, 0.5, f"${latex_str}$", ha='center', va='center', fontsize=fontsize, color='white')
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    return fig

# --- Lógica de Cálculo (Backend) ---
def calcular_polos_y_residuos(func, var):
    polos = {}
    try:
        singularidades = sympy.singularities(func, var)
        for s in singularidades:
            residuo = sympy.residue(func, var, s)
            if residuo.is_finite and residuo != 0:
                polos[s] = residuo
    except Exception:
        return {}
    return polos

def visualizar_plano_complejo(polos_residuos, polos_encerrados, tipo_contorno, params):
    fig, ax = plt.subplots(figsize=(8, 8))
    
    lim_max = 5
    if tipo_contorno == "Círculo":
        centro, radio = complex(params['centro']), float(params['radio'])
        contorno_shape = plt.Circle((centro.real, centro.imag), radio, color='cyan', fill=False, lw=2, label='Contorno C (Círculo)')
        ax.add_patch(contorno_shape)
        lim_max = max(abs(centro.real), abs(centro.imag)) + radio + 1.5
    elif tipo_contorno == "Rectángulo":
        p1 = complex(params['esquina_inf_izq'])
        p2 = complex(params['esquina_sup_der'])
        ancho = p2.real - p1.real
        alto = p2.imag - p1.imag
        contorno_shape = plt.Rectangle((p1.real, p1.imag), ancho, alto, color='#DA70D6', fill=False, lw=2, label='Contorno C (Rectángulo)')
        ax.add_patch(contorno_shape)
        lim_max = max(abs(p1.real), abs(p1.imag), abs(p2.real), abs(p2.imag)) + 1.5
    
    for polo, residuo in polos_residuos.items():
        polo_num = complex(polo.evalf())
        color = 'red' if polo in polos_encerrados else '#CCCCCC'
        label = '(Encerrado)' if polo in polos_encerrados else '(Fuera)'
        ax.scatter(polo_num.real, polo_num.imag, color=color, s=100, zorder=5)
        ax.text(polo_num.real + 0.05 * lim_max, polo_num.imag + 0.05 * lim_max, f'z={polo_num.real:.1f}+{polo_num.imag:.1f}j\n{label}', color=color, fontsize=9)
            
    ax.set_xlim(-lim_max, lim_max)
    ax.set_ylim(-lim_max, lim_max)
    ax.set_xlabel("Eje Real")
    ax.set_ylabel("Eje Imaginario")
    ax.set_title("Plano Complejo: Contorno y Polos")
    ax.axhline(0, color='gray', lw=0.5, alpha=0.7)
    ax.axvline(0, color='gray', lw=0.5, alpha=0.7)
    ax.grid(True, linestyle='--', alpha=0.2)
    ax.set_aspect('equal', adjustable='box')
    
    legend = ax.legend()
    if legend:
        legend.get_frame().set_alpha(0)
        for text in legend.get_texts():
            text.set_color('white')

    return fig

def analizar_funcion_compleja(funcion_str, tipo_contorno, centro_str, radio_str, esq_inf_izq_str, esq_sup_der_str):
    try:
        local_dict = {'z': z, 'I': sympy.I, 'pi': sympy.pi, 'exp': sympy.exp, 'sin': sympy.sin, 'cos': sympy.cos, 'sqrt': sympy.sqrt}
        funcion = sympy.sympify(funcion_str, locals=local_dict)
    except (sympy.SympifyError, SyntaxError):
        return "### Error\nLa función que ingresaste no es válida. Revisa la sintaxis.", None

    polos_y_residuos = calcular_polos_y_residuos(funcion, z)
    
    suma_residuos, polos_encerrados, params = 0, [], {}
    
    try:
        if tipo_contorno == "Círculo":
            centro = complex(centro_str.replace('i', 'j'))
            radio = float(radio_str)
            params = {'centro': centro_str, 'radio': radio_str}
            for polo, residuo in polos_y_residuos.items():
                if abs(complex(polo.evalf()) - centro) < radio:
                    polos_encerrados.append(polo)
                    suma_residuos += residuo
        elif tipo_contorno == "Rectángulo":
            p1 = complex(esq_inf_izq_str.replace('i', 'j'))
            p2 = complex(esq_sup_der_str.replace('i', 'j'))
            params = {'esquina_inf_izq': esq_inf_izq_str, 'esquina_sup_der': esq_sup_der_str}
            for polo, residuo in polos_y_residuos.items():
                polo_num = complex(polo.evalf())
                if p1.real < polo_num.real < p2.real and p1.imag < polo_num.imag < p2.imag:
                    polos_encerrados.append(polo)
                    suma_residuos += residuo
    except (ValueError, TypeError):
        return "### Error\nLos parámetros del contorno no son válidos. Revisa los números ingresados.", None

    resultados_md = f"### Análisis para `f(z) = {funcion}`\n"
    if not polos_y_residuos:
        resultados_md += "**No se encontraron polos para la función dada.**"
    else:
        resultados_md += "**Polos y Residuos Encontrados:**\n"
        for polo, residuo in polos_y_residuos.items():
            resultados_md += f"* **Polo en z = `{polo}`** → Residuo = `{sympy.simplify(residuo)}`\n"
    
    resultados_md += "\n### Teorema del Residuo\n"
    if not polos_encerrados:
        resultados_md += "**Ningún polo se encuentra dentro del contorno.**\n"
        resultado_integral = 0
    else:
        resultados_md += "**Polos encerrados por el contorno:**\n"
        for p in polos_encerrados:
            resultados_md += f"* `{p}`\n"
        resultado_integral = 2 * sympy.pi * sympy.I * suma_residuos
    
    resultados_md += f"\n**Suma de residuos encerrados:** `{sympy.simplify(suma_residuos)}`\n"
    resultados_md += f"#### Resultado de la integral ∮f(z)dz = `{sympy.simplify(resultado_integral)}`"

    fig = visualizar_plano_complejo(polos_y_residuos, polos_encerrados, tipo_contorno, params)
    
    return resultados_md, fig

# --- Interfaz Gráfica con Gradio (UI) ---

def actualizar_contorno(tipo):
    if tipo == "Círculo":
        return gr.update(visible=True), gr.update(visible=False)
    else:
        return gr.update(visible=False), gr.update(visible=True)

theme = gr.themes.Default(
    primary_hue="cyan",
    secondary_hue="purple",
)

with gr.Blocks(theme=theme) as iface:
    gr.Markdown("# 🌀 Calculadora de Residuos y Contornos PRO 🌀")
    
    with gr.Tabs() as tabs:
        with gr.TabItem("Calculadora", id=0):
            # ... (Sin cambios)...
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 1. Define tu Función")
                    funcion_input = gr.Textbox(label="Función f(z)", value="exp(z) / (z - 2)**3", info="Usa sintaxis de Python. Ej: z**3, exp(z), sin(z), pi, I")
                    
                    gr.Markdown("### 2. Elige el Contorno")
                    contorno_tipo_input = gr.Radio(["Círculo", "Rectángulo"], label="Tipo de Contorno", value="Círculo")
                    
                    with gr.Group(visible=True) as grupo_circulo:
                        centro_input = gr.Textbox(label="Centro (ej: 0+0j)", value="0+0j")
                        radio_input = gr.Slider(0.1, 10.0, value=3.0, label="Radio", step=0.1)
                    
                    with gr.Group(visible=False) as grupo_rectangulo:
                        esq_inf_izq_input = gr.Textbox(label="Esquina Inferior Izquierda (ej: -1-1j)", value="-1-1j")
                        esq_sup_der_input = gr.Textbox(label="Esquina Superior Derecha (ej: 1+1j)", value="1+1j")

                    btn_calcular = gr.Button("Calcular Análisis", variant="primary")

                with gr.Column(scale=2):
                    gr.Markdown("### 3. Resultados")
                    resultados_output = gr.Markdown(label="Resultados del Análisis")
                    plot_output = gr.Plot(label="Plano Complejo")

        with gr.TabItem("Guía y Ejemplos", id=1):
            gr.Markdown("""
            ## ¿Cómo Usar la Calculadora?
            1.  **Escribe tu función:** En la pestaña "Calculadora", ingresa tu función usando `z` como la variable compleja. Puedes usar `I` para la unidad imaginaria, `pi` para π, y funciones como `exp()`, `sin()`, `cos()`.
            2.  **Selecciona un contorno:** Elige entre "Círculo" o "Rectángulo".
            3.  **Define sus parámetros:** Ajusta el centro/radio o las esquinas según tu elección.
            4.  **Calcula:** Presiona el botón para ver el análisis de residuos, el resultado de la integral y la gráfica. **La calculadora maneja automáticamente polos simples y de orden superior.**

            ## Ejemplos para Probar
            Haz clic en un ejemplo para cargarlo automáticamente en la calculadora.
            """)
            gr.Examples(
                examples=[
                    ["exp(z) / (z - 2)**3", "Círculo", "0+0j", "3.0", "-1-1j", "1+1j"],
                    ["1 / (z**2 + 4)", "Círculo", "0+0j", "3.0", "-1-1j", "1+1j"],
                    ["z**2 / (z-1)", "Círculo", "0+0j", "0.5", "-1-1j", "1+1j"],
                    ["1 / sin(z)", "Rectángulo", "0+0j", "4.0", "-4-1j", "4+1j"],
                ],
                inputs=[funcion_input, contorno_tipo_input, centro_input, radio_input, esq_inf_izq_input, esq_sup_der_input],
                label="Haz clic para cargar un ejemplo (el primero tiene un polo de orden 3)"
            )

        with gr.TabItem("Teoría", id=2):
            gr.Markdown("## Teorema del Residuo")
            gr.Markdown(
                "El **Teorema del Residuo** establece que la integral de contorno de una función es igual a $2\pi i$ por la suma de los residuos de sus polos encerrados en dicho contorno."
            )
            gr.Plot(value=render_latex_formula(r"\oint_C f(z) dz = 2\pi i \sum_{k=1}^{n} \mathrm{Res}(f, z_k)"), label="Fórmula del Teorema del Residuo")
            
            gr.Markdown("---") # Separador
            gr.Markdown("## Cálculo de Residuos")
            gr.Markdown("El método para calcular el residuo depende del orden del polo en $z_0$:")

            gr.Markdown("### 1. Polo Simple (Orden 1)")
            gr.Markdown("Para un polo simple, el residuo se calcula con el siguiente límite:")
            gr.Plot(value=render_latex_formula(r"\mathrm{Res}(f, z_0) = \lim_{z \to z_0} (z - z_0) f(z)", fontsize=20), label="Fórmula para Polo Simple")

            gr.Markdown("### 2. Polo de Orden *m*")
            gr.Markdown("Para un polo de orden *m* > 1, la fórmula general implica una derivada:")
            gr.Plot(value=render_latex_formula(r"\mathrm{Res}(f, z_0) = \frac{1}{(m-1)!} \lim_{z \to z_0} \frac{d^{m-1}}{dz^{m-1}} \left[ (z - z_0)^m f(z) \right]", fontsize=20), label="Fórmula para Polo de Orden m")
            
            gr.Markdown("**¡Nuestra calculadora aplica automáticamente la fórmula correcta para cada polo que encuentra!**")


    # Lógica de eventos de la UI
    contorno_tipo_input.change(fn=actualizar_contorno, inputs=contorno_tipo_input, outputs=[grupo_circulo, grupo_rectangulo], show_progress=False)
    btn_calcular.click(
        fn=analizar_funcion_compleja,
        inputs=[funcion_input, contorno_tipo_input, centro_input, radio_input, esq_inf_izq_input, esq_sup_der_input],
        outputs=[resultados_output, plot_output]
    )

if __name__ == "__main__":
    iface.launch()
