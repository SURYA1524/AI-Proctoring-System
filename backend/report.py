from fpdf import FPDF
import matplotlib.pyplot as plt

def generate_report(events, scores):
    plt.plot(scores)
    plt.title("Suspicion Score Over Time")
    plt.xlabel("Time")
    plt.ylabel("Score")
    plt.savefig("score_graph.png")
    plt.close()

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, "AI PROCTORING REPORT", ln=True)

    for e in events:
        pdf.cell(200, 8, f"{e['time']} - {e['event']}", ln=True)

    pdf.image("score_graph.png", x=10, y=100, w=180)
    pdf.output("cheating_report.pdf")