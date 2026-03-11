const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, LevelFormat,
  TabStopType, TabStopPosition,
  TableOfContents, HeadingLevel, BorderStyle, WidthType, ShadingType,
  PageNumber, PageBreak, SectionType
} = require("docx");

// ── Constants ──────────────────────────────────────────────────────────────
const FONT = "Times New Roman";
const PAGE_WIDTH = 11906; // A4
const PAGE_HEIGHT = 16838;
const MARGIN = 1440; // 1 inch
const CONTENT_WIDTH = PAGE_WIDTH - 2 * MARGIN;
const thinBorder = { style: BorderStyle.SINGLE, size: 1, color: "000000" };
const borders = { top: thinBorder, bottom: thinBorder, left: thinBorder, right: thinBorder };
const cellMargins = { top: 60, bottom: 60, left: 100, right: 100 };

// ── Helpers ────────────────────────────────────────────────────────────────
const txt = (text, opts = {}) => new TextRun({ text, font: FONT, size: opts.size || 24, bold: opts.bold, italics: opts.italics, underline: opts.underline });
const emptyLine = () => new Paragraph({ spacing: { before: 0, after: 0 }, children: [new TextRun({ text: "", font: FONT, size: 24 })] });
const centered = (children, opts = {}) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: opts.before || 0, after: opts.after || 0 }, children });
const justified = (children, opts = {}) => new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { before: opts.before || 0, after: opts.after || 120, line: opts.line || 360 }, children, indent: opts.indent });
const leftPara = (children, opts = {}) => new Paragraph({ alignment: AlignmentType.LEFT, spacing: { before: opts.before || 0, after: opts.after || 120, line: opts.line || 360 }, children, indent: opts.indent });

const heading1 = (text) => new Paragraph({
  heading: HeadingLevel.HEADING_1, alignment: AlignmentType.CENTER,
  spacing: { before: 240, after: 240 },
  children: [new TextRun({ text: text.toUpperCase(), font: FONT, size: 32, bold: true })]
});
const heading2 = (text) => new Paragraph({
  heading: HeadingLevel.HEADING_2, spacing: { before: 200, after: 160 },
  children: [new TextRun({ text, font: FONT, size: 28, bold: true })]
});
const heading3 = (text) => new Paragraph({
  heading: HeadingLevel.HEADING_3, spacing: { before: 160, after: 120 },
  children: [new TextRun({ text, font: FONT, size: 26, bold: true, italics: true })]
});
const pageBreak = () => new Paragraph({ children: [new PageBreak()] });

function makeTableRow(cells, isHeader = false) {
  return new TableRow({
    tableHeader: isHeader,
    children: cells.map(c => {
      const d = typeof c === "string" ? { text: c } : c;
      return new TableCell({
        borders, width: { size: d.width || 2000, type: WidthType.DXA },
        shading: isHeader ? { fill: "D9E2F3", type: ShadingType.CLEAR } : undefined,
        margins: cellMargins, verticalAlign: "center",
        children: [new Paragraph({
          alignment: d.align || (isHeader ? AlignmentType.CENTER : AlignmentType.LEFT),
          children: [new TextRun({ text: d.text || "", font: FONT, size: 22, bold: isHeader || d.bold })]
        })]
      });
    })
  });
}

function makeTable(headers, rows, colWidths) {
  const widths = colWidths || headers.map(() => Math.floor(CONTENT_WIDTH / headers.length));
  const totalWidth = widths.reduce((a, b) => a + b, 0);
  return new Table({
    width: { size: totalWidth, type: WidthType.DXA }, columnWidths: widths,
    rows: [
      makeTableRow(headers.map((h, i) => ({ text: h, width: widths[i] })), true),
      ...rows.map(row => makeTableRow(row.map((c, i) => {
        const cd = typeof c === "string" ? { text: c } : c;
        return { ...cd, width: widths[i] };
      })))
    ]
  });
}

// ── Title Page ─────────────────────────────────────────────────────────────
function titlePage() {
  return [
    emptyLine(),
    centered([txt("ADVANCED DEEP NEURAL NETWORK FRAMEWORK FOR EARLY PLANT DISEASE IDENTIFICATION AND PRECISION FARMING", { size: 32, bold: true })], { before: 600, after: 300 }),
    emptyLine(),
    centered([txt("A Main Project submitted in partial fulfilment of the requirements", { size: 24 })]),
    centered([txt("for the award of the degree of", { size: 24 })]),
    emptyLine(),
    centered([txt("BACHELOR OF TECHNOLOGY", { size: 28, bold: true })], { before: 100 }),
    centered([txt("In", { size: 24 })]),
    centered([txt("INFORMATION TECHNOLOGY", { size: 28, bold: true })], { after: 200 }),
    emptyLine(),
    centered([txt("Submitted by", { size: 24 })]),
    emptyLine(),
    centered([txt("1.  [Student Name 1] ([Reg.No])                    2.  [Student Name 2] ([Reg.No])", { size: 24 })]),
    centered([txt("3.  [Student Name 3] ([Reg.No])                    4.  [Student Name 4] ([Reg.No])", { size: 24 })]),
    emptyLine(), emptyLine(),
    centered([txt("Under the esteemed guidance of", { size: 24 })]),
    emptyLine(),
    centered([txt("[Guide Name]", { size: 24, bold: true })]),
    centered([txt("Assistant Professor", { size: 24 })]),
    emptyLine(), emptyLine(),
    centered([txt("DEPARTMENT OF INFORMATION TECHNOLOGY", { size: 24, bold: true })], { before: 200 }),
    centered([txt("VISHNU INSTITUTE OF TECHNOLOGY", { size: 26, bold: true })]),
    centered([txt("(Autonomous)", { size: 22 })]),
    centered([txt("(Approved by AICTE, Accredited by NBA & NAAC and permanently affiliated to JNTU Kakinada)", { size: 20 })]),
    centered([txt("BHIMAVARAM - 534202", { size: 24, bold: true })]),
    centered([txt("2025 - 2026", { size: 26, bold: true })], { before: 100 }),
  ];
}

// ── Certificate ────────────────────────────────────────────────────────────
function certificatePage() {
  return [
    pageBreak(),
    emptyLine(),
    centered([txt("VISHNU INSTITUTE OF TECHNOLOGY", { size: 26, bold: true })], { before: 100 }),
    centered([txt("(Autonomous)", { size: 22 })]),
    centered([txt("(Approved by AICTE, Accredited by NBA & NAAC and permanently affiliated to JNTU Kakinada)", { size: 20 })]),
    centered([txt("BHIMAVARAM - 534202", { size: 24, bold: true })]),
    centered([txt("2025 - 2026", { size: 24 })]),
    centered([txt("DEPARTMENT OF INFORMATION TECHNOLOGY", { size: 24, bold: true })], { before: 100, after: 200 }),
    emptyLine(),
    centered([txt("CERTIFICATE", { size: 32, bold: true, underline: {} })], { before: 100, after: 300 }),
    emptyLine(),
    justified([
      txt("This is to certify that the project entitled ", { size: 24 }),
      txt('"Advanced Deep Neural Network Framework for Early Plant Disease Identification and Precision Farming"', { size: 24, bold: true }),
      txt(", is being submitted by [Student Name 1], [Student Name 2], [Student Name 3] and [Student Name 4], bearing the REGD.NOS: [Reg.No], [Reg.No], [Reg.No] and [Reg.No] submitted in partial fulfilment for the award of the degree of ", { size: 24 }),
      txt("BACHELOR OF TECHNOLOGY", { size: 24, bold: true }),
      txt(" in ", { size: 24 }),
      txt("INFORMATION TECHNOLOGY", { size: 24, bold: true }),
      txt(" is a record of bonafide work carried out by them under my guidance and supervision during the academic year 2025-2026 and it has been found worthy of acceptance according to the requirements of university.", { size: 24 }),
    ]),
    emptyLine(), emptyLine(), emptyLine(),
    new Paragraph({
      spacing: { before: 400 },
      tabStops: [{ type: TabStopType.RIGHT, position: TabStopPosition.MAX }],
      children: [txt("Internal Guide", { size: 24 }), new TextRun({ text: "\tHead of the Department", font: FONT, size: 24 })],
    }),
    new Paragraph({
      tabStops: [{ type: TabStopType.RIGHT, position: TabStopPosition.MAX }],
      children: [txt("[Guide Name]", { size: 24 }), new TextRun({ text: "\tMrs. M Srilakshmi", font: FONT, size: 24 })],
    }),
    new Paragraph({
      tabStops: [{ type: TabStopType.RIGHT, position: TabStopPosition.MAX }],
      children: [txt("[Designation]", { size: 24 }), new TextRun({ text: "\tProfessor", font: FONT, size: 24 })],
    }),
    emptyLine(), emptyLine(),
    leftPara([txt("External Examiner", { size: 24 })]),
  ];
}

// ── Acknowledgement ────────────────────────────────────────────────────────
function acknowledgementPage() {
  return [
    pageBreak(),
    centered([txt("ACKNOWLEDGEMENT", { size: 32, bold: true, underline: {} })], { before: 400, after: 400 }),
    emptyLine(),
    justified([txt("It is nature and inevitable that the thoughts and ideas of other people tend to drift in to the subconscious due to various human parameters, where one feels acknowledge the help and guidance derived from others. We acknowledge each of those who have contributed for the fulfilment of this project.", { size: 24 })]),
    justified([txt("We take the opportunity to express our sincere gratitude to Dr. M. Venu, Principal, Vishnu Institute of Technology, Bhimavaram whose guidance from time to time helped us to complete this project successfully.", { size: 24 })]),
    justified([txt("We are very much thankful to Mrs. M. Srilakshmi, Vice Principal and Head of the Department, INFORMATION TECHNOLOGY for her continuous and unrelenting support and guidance. We thank and acknowledge our gratitude to her valuable guidance and support expended to us right from the conception of the idea to the completion of this project.", { size: 24 })]),
    justified([txt("We are very much thankful to [Guide Name], [Designation], our internal guide whose guidance from time to time helped us to complete this project successfully.", { size: 24 })]),
    emptyLine(), emptyLine(),
    leftPara([txt("[Student Name 1] ([Reg.No])", { size: 24 })], { after: 40 }),
    leftPara([txt("[Student Name 2] ([Reg.No])", { size: 24 })], { after: 40 }),
    leftPara([txt("[Student Name 3] ([Reg.No])", { size: 24 })], { after: 40 }),
    leftPara([txt("[Student Name 4] ([Reg.No])", { size: 24 })], { after: 40 }),
    emptyLine(),
    leftPara([txt("Project Associates", { size: 24, bold: true })]),
  ];
}

// ── Abstract ───────────────────────────────────────────────────────────────
function abstractPage() {
  return [
    pageBreak(),
    centered([txt("ABSTRACT", { size: 32, bold: true, underline: {} })], { before: 400, after: 400 }),
    emptyLine(),
    justified([txt("Plant diseases pose a significant threat to global agriculture, affecting crop yield and quality, and leading to substantial economic losses. Traditional methods of plant disease identification rely on manual visual inspection by domain experts, which is time-consuming, subjective, and often inaccessible to small-scale farmers. This project presents an Advanced Deep Neural Network Framework for Early Plant Disease Identification and Precision Farming, leveraging a custom-designed four-block Convolutional Neural Network (CNN) architecture built with PyTorch. The model is trained on the Plant Village dataset comprising 61,486 images across 39 disease classes spanning 14 plant species, achieving approximately 97% training accuracy and 99% validation accuracy. The framework is deployed as a responsive Flask web application that allows users to upload leaf images via drag-and-drop, camera capture, or voice commands. The system provides real-time disease diagnosis with confidence scoring, severity classification (Healthy, Mild, Moderate, Critical), and recommends targeted agricultural supplements with direct purchase links. Additional features include multilingual support for 10 Indian languages through Google Translator integration, text-to-speech capabilities for accessibility, and a community-based alert system with geolocation mapping for regional disease monitoring. This comprehensive solution bridges the gap between advanced deep learning research and practical precision farming, empowering farmers with accessible, accurate, and actionable plant health diagnostics.", { size: 24 })]),
  ];
}

// ── TOC ────────────────────────────────────────────────────────────────────
function tocPage() {
  return [
    pageBreak(),
    centered([txt("TABLE OF CONTENTS", { size: 32, bold: true, underline: {} })], { before: 200, after: 300 }),
    emptyLine(),
    new TableOfContents("Table of Contents", { hyperlink: true, headingStyleRange: "1-3" }),
  ];
}

// ── List of Tables ─────────────────────────────────────────────────────────
function listOfTablesPage() {
  return [
    pageBreak(),
    centered([txt("LIST OF TABLES", { size: 32, bold: true, underline: {} })], { before: 200, after: 300 }),
    emptyLine(),
    makeTable(["Sl. No", "Table Name", "Page No."],
      [["1","Table 2.1 Hardware Requirements",""],["2","Table 2.2 Software Requirements",""],["3","Table 5.1 Technologies Used",""],["4","Table 5.2 Disease Classes and Indices",""],["5","Table 5.3 Database Schema - disease_alerts",""],["6","Table 5.4 Database Schema - alert_subscriptions",""],["7","Table 6.1 Test Cases",""]],
      [1200, 6500, 1326]),
  ];
}

// ── List of Figures ────────────────────────────────────────────────────────
function listOfFiguresPage() {
  return [
    pageBreak(),
    centered([txt("LIST OF FIGURES", { size: 32, bold: true, underline: {} })], { before: 200, after: 300 }),
    emptyLine(),
    makeTable(["Sl. No", "Figure Name", "Page No."],
      [["1","Figure 3.1 System Architecture",""],["2","Figure 3.2 CNN Model Architecture",""],["3","Figure 3.3 Data Flow Diagram",""],["4","Figure 3.4 Use Case Diagram",""],["5","Figure 3.5 Class Diagram",""],["6","Figure 3.6 Sequence Diagram",""],["7","Figure 3.7 ER Diagram",""],["8","Figure 5.1 Home Page Screenshot",""],["9","Figure 5.2 Upload Page Screenshot",""],["10","Figure 5.3 Results Page Screenshot",""],["11","Figure 5.4 Marketplace Page Screenshot",""],["12","Figure 5.5 Community Alerts Page Screenshot",""]],
      [1200, 6500, 1326]),
  ];
}

// ── Chapter 1: Introduction ────────────────────────────────────────────────
function chapter1() {
  return [
    heading1("Chapter 1: Introduction"),
    heading2("1.1 Background"),
    justified([txt("Agriculture is the backbone of the global economy, sustaining livelihoods for billions of people worldwide. Plant diseases are one of the most critical threats to agricultural productivity, causing estimated annual crop losses of 20-40% globally according to the Food and Agriculture Organization (FAO). Early and accurate detection of plant diseases is essential for implementing timely interventions and minimizing crop damage.", { size: 24 })]),
    justified([txt("Traditional methods of disease identification rely heavily on manual visual inspection by trained agronomists or plant pathologists. This approach is inherently limited by the availability of experts, subjective assessment criteria, and the inability to scale across large agricultural regions. Furthermore, many diseases present subtle visual symptoms in their early stages that are difficult even for experienced professionals to distinguish.", { size: 24 })]),
    justified([txt("The advent of deep learning and computer vision has opened new frontiers in automated plant disease detection. Convolutional Neural Networks (CNNs) have demonstrated remarkable success in image classification tasks, including medical imaging, autonomous vehicles, and agricultural applications. By training CNNs on large datasets of diseased and healthy plant leaf images, it is possible to build systems that can classify plant diseases with accuracy rivaling or surpassing human experts.", { size: 24 })]),
    heading2("1.2 Problem Statement"),
    justified([txt("Small-scale and marginal farmers, particularly in developing countries like India, often lack access to agricultural extension services and plant disease diagnosis facilities. By the time a disease is visually identified and expert advice is sought, significant crop damage may have already occurred. There is a pressing need for an accessible, accurate, and real-time plant disease detection system that can be used by farmers with minimal technical expertise, using just a smartphone camera.", { size: 24 })]),
    heading2("1.3 Objectives"),
    justified([txt("The primary objectives of this project are:", { size: 24 })]),
    ...[
      "To design and implement a custom CNN architecture capable of classifying 39 different plant disease classes across 14 plant species with high accuracy.",
      "To develop a user-friendly web application using Flask that allows farmers to upload or capture leaf images for instant disease diagnosis.",
      "To provide actionable recommendations including disease descriptions, prevention steps, and targeted supplement/fertilizer suggestions with purchase links.",
      "To implement multilingual support covering 10 Indian languages for wider accessibility among diverse farming communities.",
      "To integrate voice-based interaction (speech recognition and text-to-speech) for accessibility.",
      "To build a community-based alert system for regional disease monitoring and early warning.",
    ].map((o, i) => justified([txt(`${i+1}. ${o}`, { size: 24 })], { indent: { left: 360 } })),
    heading2("1.4 Scope of the Project"),
    justified([txt("This project encompasses the complete lifecycle of a deep learning-based plant disease detection system, from model training to web deployment. The system supports identification of 39 classes including 26 disease conditions and 12 healthy states plus one background class, covering 14 plant species: Apple, Blueberry, Cherry, Corn (Maize), Grape, Orange, Peach, Pepper Bell, Potato, Raspberry, Soybean, Squash, Strawberry, and Tomato. The web application provides a responsive interface accessible from both desktop and mobile devices.", { size: 24 })]),
    heading2("1.5 Organization of the Report"),
    justified([txt("This report is organized into seven chapters. Chapter 1 provides the introduction and background. Chapter 2 presents the system analysis including requirements and feasibility. Chapter 3 details the system design with architectural diagrams. Chapter 4 describes each module in depth. Chapter 5 covers the implementation with technologies and code samples. Chapter 6 discusses the testing methodology and results. Chapter 7 concludes the report with a summary and future scope.", { size: 24 })]),
  ];
}

// ── Chapter 2: System Analysis ─────────────────────────────────────────────
function chapter2() {
  return [
    pageBreak(), heading1("Chapter 2: System Analysis"),
    heading2("2.1 Hardware and Software Requirements"),
    heading3("2.1.1 Hardware Requirements"),
    justified([txt("The following table lists the minimum hardware requirements for developing, training, and deploying the plant disease detection system:", { size: 24 })]),
    makeTable(["Component", "Minimum Requirement", "Recommended"],
      [["Processor","Intel Core i5 / AMD Ryzen 5","Intel Core i7 / AMD Ryzen 7"],["RAM","8 GB","16 GB"],["Storage","10 GB free space","20 GB SSD"],["GPU (Training)","Not required (CPU training)","NVIDIA GPU with CUDA"],["Display","1366 x 768","1920 x 1080"],["Network","Broadband internet","High-speed internet"],["Camera","Any webcam / smartphone","HD webcam / smartphone"]], [2500, 3263, 3263]),
    centered([txt("Table 2.1: Hardware Requirements", { size: 20, italics: true })], { before: 60, after: 200 }),
    heading3("2.1.2 Software Requirements"),
    makeTable(["Software", "Version", "Purpose"],
      [["Python","3.8+ (3.10+ recommended)","Programming language"],["Flask",">= 2.3","Web application framework"],["PyTorch",">= 2.0","Deep learning framework"],["TorchVision",">= 0.15","Image transformations"],["Pillow",">= 10.0","Image processing"],["Pandas",">= 2.0","CSV data handling"],["NumPy",">= 1.24","Numerical computing"],["Deep Translator",">= 1.11","Translation"],["Gunicorn",">= 21","Production server"],["Bootstrap 5","5.x","CSS framework"],["Font Awesome","6.4.0","Icon library"],["Web Browser","Chrome/Firefox/Edge","Client interface"]], [2500, 3263, 3263]),
    centered([txt("Table 2.2: Software Requirements", { size: 20, italics: true })], { before: 60, after: 200 }),

    heading2("2.2 Existing System and Its Disadvantages"),
    justified([txt("The existing approaches to plant disease detection include:", { size: 24 })]),
    justified([txt("Manual Visual Inspection: Farmers and agricultural officers inspect crops visually to identify diseases. This method is subjective, time-consuming, and requires domain expertise that is not readily available in rural areas.", { size: 24 })], { indent: { left: 360 } }),
    justified([txt("Laboratory Testing: Plant samples are collected and sent to laboratories for pathological analysis. While accurate, this process takes days to weeks and is expensive, making it impractical for real-time decision making.", { size: 24 })], { indent: { left: 360 } }),
    justified([txt("Existing Mobile Applications: Some applications exist but are limited in the number of diseases they can detect, lack multilingual support, and do not provide comprehensive treatment recommendations.", { size: 24 })], { indent: { left: 360 } }),
    emptyLine(),
    justified([txt("Disadvantages of Existing Systems:", { size: 24, bold: true })]),
    ...["High dependency on human expertise and availability of plant pathologists.","Slow turnaround time for lab-based diagnosis.","Limited scalability across large farming areas.","High cost of expert consultation and lab testing.","No real-time community alerting mechanism.","Language barriers for non-English speaking farmers."].map((d,i) => justified([txt(`${i+1}. ${d}`, { size: 24 })], { indent: { left: 360 } })),

    heading2("2.3 Proposed System and Its Advantages"),
    justified([txt("The proposed system is an Advanced Deep Neural Network Framework that leverages a custom four-block CNN architecture to automatically detect and classify 39 plant disease conditions from leaf images. The system is deployed as a responsive Flask web application accessible from any device with a web browser.", { size: 24 })]),
    emptyLine(),
    justified([txt("Advantages of the Proposed System:", { size: 24, bold: true })]),
    ...["Instant disease diagnosis - results in seconds, not days.","High accuracy - approximately 97% training and 99% validation accuracy.","Wide coverage - 39 classes across 14 plant species.","User-friendly interface with drag-and-drop, camera capture, and voice commands.","Multilingual support for 10 Indian languages.","Actionable recommendations with supplement/fertilizer purchase links.","Community alert system with geolocation-based disease monitoring.","Confidence scoring with severity classification for informed decisions.","Accessible on any device - no specialized hardware required.","Cost-effective - free to use, reducing dependence on expensive consultations."].map((a,i) => justified([txt(`${i+1}. ${a}`, { size: 24 })], { indent: { left: 360 } })),

    heading2("2.4 Feasibility Study"),
    heading3("2.4.1 Technical Feasibility"),
    justified([txt("The project is technically feasible as it leverages well-established technologies. PyTorch provides a mature deep learning framework for CNN implementation. Flask is a lightweight and proven web framework for Python applications. The Plant Village dataset with 61,486 labeled images provides sufficient training data. The model runs on CPU, eliminating the need for expensive GPU hardware in production. All required libraries are open-source and freely available.", { size: 24 })]),
    heading3("2.4.2 Economic Feasibility"),
    justified([txt("The project uses entirely open-source technologies with no licensing costs. The application can be hosted on free or low-cost cloud platforms such as Heroku, Render, or Railway. The CPU-only inference requirement minimizes server costs. The model file is approximately 100 MB, requiring minimal storage. The overall development and deployment cost is minimal, making it economically viable for widespread adoption.", { size: 24 })]),
    heading3("2.4.3 Operational Feasibility"),
    justified([txt("The application is designed with simplicity in mind. Users need only a web browser and a camera to use the system. The drag-and-drop upload, camera capture, and voice command features ensure accessibility for users with varying levels of technical expertise. Multilingual support in 10 Indian languages addresses the linguistic diversity of the target user base. The responsive design works seamlessly on mobile devices, which are the primary computing devices for most farmers.", { size: 24 })]),
  ];
}

// ── Chapter 3: System Design ───────────────────────────────────────────────
function chapter3() {
  return [
    pageBreak(), heading1("Chapter 3: System Design"),
    heading2("3.1 System Architecture"),
    justified([txt("The system follows a three-tier architecture comprising the Presentation Layer (Frontend), Application Layer (Flask Backend), and Data Layer (Model + CSV/Database). The user interacts with the web interface to upload leaf images. The Flask backend processes the image through the trained CNN model, queries the disease and supplement databases, and returns the results to the frontend for display.", { size: 24 })]),
    justified([txt("[Insert Figure 3.1: System Architecture Diagram here]", { size: 24, italics: true, bold: true })]),
    emptyLine(),
    justified([txt("The high-level data flow is: User uploads image through web browser, image is saved to server (static/uploads/), PIL opens and resizes image to 224x224, torchvision transforms convert to tensor, CNN model performs forward pass producing 39-class probability distribution, argmax selects predicted class, disease and supplement information is retrieved from CSV files, and results are rendered back to the user.", { size: 24 })]),

    heading2("3.2 CNN Model Architecture"),
    justified([txt("The core of the system is a custom four-block Convolutional Neural Network designed for multi-class image classification. Each block consists of two convolution layers with ReLU activation and Batch Normalization, followed by Max Pooling for spatial downsampling.", { size: 24 })]),
    emptyLine(),
    justified([txt("Architecture Details:", { size: 24, bold: true })]),
    ...["Block 1: Conv2d(3, 32, kernel=3, padding=1) -> ReLU -> BatchNorm2d(32) -> Conv2d(32, 32) -> ReLU -> BatchNorm2d(32) -> MaxPool2d(2)",
      "Block 2: Conv2d(32, 64) -> ReLU -> BatchNorm2d(64) -> Conv2d(64, 64) -> ReLU -> BatchNorm2d(64) -> MaxPool2d(2)",
      "Block 3: Conv2d(64, 128) -> ReLU -> BatchNorm2d(128) -> Conv2d(128, 128) -> ReLU -> BatchNorm2d(128) -> MaxPool2d(2)",
      "Block 4: Conv2d(128, 256) -> ReLU -> BatchNorm2d(256) -> Conv2d(256, 256) -> ReLU -> BatchNorm2d(256) -> MaxPool2d(2)",
      "Classifier: Flatten(50176) -> Dropout(0.4) -> Linear(1024) -> ReLU -> Dropout(0.4) -> Linear(39)"
    ].map(l => justified([txt(l, { size: 22 })], { indent: { left: 360 } })),
    emptyLine(),
    justified([txt("[Insert Figure 3.2: CNN Model Architecture Diagram here]", { size: 24, italics: true, bold: true })]),
    emptyLine(),
    justified([txt("The channel progression (3 -> 32 -> 64 -> 128 -> 256) follows the standard practice of doubling channels while halving spatial dimensions through max pooling. Batch normalization after each convolution stabilizes training. Dropout (0.4) in the classifier prevents overfitting. The model accepts 224x224 RGB input and outputs a 39-dimensional probability vector.", { size: 24 })]),

    heading2("3.3 Data Flow Diagram"),
    justified([txt("The Data Flow Diagram (DFD) illustrates how data moves through the system from user input to result display.", { size: 24 })]),
    justified([txt("Level 0 DFD (Context Diagram):", { size: 24, bold: true })]),
    justified([txt("External Entity: User (Farmer) provides leaf image input and receives disease diagnosis output. The system processes the image through the CNN model and returns results with supplement recommendations.", { size: 24 })]),
    justified([txt("Level 1 DFD:", { size: 24, bold: true })]),
    ...["Process 1.0 - Image Upload: User uploads image via drag-drop, file browse, or camera capture. Image is validated and saved to server storage.",
      "Process 2.0 - Image Preprocessing: PIL opens image, resizes to 224x224, converts to tensor using torchvision transforms.",
      "Process 3.0 - Disease Prediction: Preprocessed tensor fed through CNN model. Softmax produces probability distribution. Top predictions extracted.",
      "Process 4.0 - Result Assembly: Disease info and supplement data queried from CSV DataFrames. Severity level determined.",
      "Process 5.0 - Result Display: Complete results rendered in web interface with translation and voice output options."
    ].map(p => justified([txt(p, { size: 24 })], { indent: { left: 360 } })),
    justified([txt("[Insert Figure 3.3: Data Flow Diagram here]", { size: 24, italics: true, bold: true })]),

    heading2("3.4 Use Case Diagram"),
    justified([txt("The Use Case Diagram identifies the primary actors and their interactions with the system.", { size: 24 })]),
    justified([txt("Actors:", { size: 24, bold: true })]),
    ...["Farmer/User: Primary actor who uploads images, views results, uses voice commands, changes language, reports alerts, and browses supplements.",
      "System (CNN Model): Processes images and generates predictions.",
      "External Services: Google Translator API, OpenStreetMap Nominatim API."
    ].map((a,i) => justified([txt(`${i+1}. ${a}`, { size: 24 })], { indent: { left: 360 } })),
    justified([txt("Use Cases:", { size: 24, bold: true })]),
    ...["UC1: Upload Leaf Image (drag-drop, file browse, camera capture)","UC2: View Disease Diagnosis (disease name, description, confidence, severity)","UC3: Use Voice Commands (speech recognition for navigation and search)","UC4: Change Language (select from 10 Indian languages)","UC5: Listen to Results (text-to-speech output)","UC6: View Supplement Recommendations","UC7: Browse Supplement Marketplace","UC8: Report Community Alert (with geolocation)","UC9: Subscribe to Regional Alerts"
    ].map(u => justified([txt(u, { size: 24 })], { indent: { left: 360 } })),
    justified([txt("[Insert Figure 3.4: Use Case Diagram here]", { size: 24, italics: true, bold: true })]),

    heading2("3.5 Class Diagram"),
    justified([txt("The Class Diagram represents the key classes and their relationships in the system.", { size: 24 })]),
    ...["CNN_Model class: Inherits from nn.Module. Contains conv_block1 through conv_block4, dense layers, and output layer. Methods: __init__(), forward(xb), prediction(). 39 output classes with idx_to_classes mapping.",
      "FlaskApp class: Manages routes (/index, /submit, /market, /alerts), loads model at startup, reads disease_info and supplement_info DataFrames. Methods: predict_image(), get_severity(), get_supplements().",
      "DiseaseInfo class: Attributes - disease_name, description, possible_steps, image_url. Loaded from disease_info.csv.",
      "SupplementInfo class: Attributes - supplement_name, supplement_image, buy_link. Loaded from supplement_info.csv.",
      "AlertSystem class: Manages community disease alerts. Database connection (SQLite/Supabase). Methods: create_alert(), get_alerts(), subscribe()."
    ].map(c => justified([txt(c, { size: 24 })], { indent: { left: 360 } })),
    justified([txt("[Insert Figure 3.5: Class Diagram here]", { size: 24, italics: true, bold: true })]),

    heading2("3.6 Sequence Diagram"),
    justified([txt("The Sequence Diagram illustrates the interaction flow for the disease prediction use case.", { size: 24 })]),
    justified([txt("Sequence: Disease Prediction Flow", { size: 24, bold: true })]),
    ...["User opens web application in browser.","User uploads leaf image (drag-drop / file browse / camera capture).","Browser sends POST request with image to /submit endpoint.","Flask saves image to static/uploads/ directory.","Flask opens image with PIL and resizes to 224x224.","Image converted to PyTorch tensor using TF.to_tensor().","Tensor passed through CNN model forward pass.","Softmax produces 39-class probability distribution.","Top-3 predictions with confidence scores extracted.","Disease info and supplement data queried from CSV DataFrames.","Results rendered in submit.html template and returned to browser.","User views diagnosis with confidence, severity, and recommendations."
    ].map((s,i) => justified([txt(`${i+1}. ${s}`, { size: 24 })], { indent: { left: 360 } })),
    justified([txt("[Insert Figure 3.6: Sequence Diagram here]", { size: 24, italics: true, bold: true })]),

    heading2("3.7 ER Diagram"),
    justified([txt("The Entity-Relationship Diagram represents the database schema for the community alerts system.", { size: 24 })]),
    justified([txt("Entities:", { size: 24, bold: true })]),
    justified([txt("1. disease_alerts: id (PK), disease_name, disease_index, severity, confidence, latitude, longitude, region_name, image_url, description, reported_by, created_at.", { size: 24 })], { indent: { left: 360 } }),
    justified([txt("2. alert_subscriptions: id (PK), email, region_name. UNIQUE constraint on (email, region_name).", { size: 24 })], { indent: { left: 360 } }),
    justified([txt("Relationship: region_name links alerts to subscriptions for regional notifications.", { size: 24 })]),
    justified([txt("[Insert Figure 3.7: ER Diagram here]", { size: 24, italics: true, bold: true })]),
  ];
}

// ── Chapter 4: Module Descriptions ─────────────────────────────────────────
function chapter4() {
  return [
    pageBreak(), heading1("Chapter 4: Module Descriptions"),
    heading2("4.1 Image Upload Module"),
    justified([txt("The Image Upload Module provides three methods for users to submit leaf images for analysis. The module is implemented in the index.html template with supporting JavaScript.", { size: 24 })]),
    justified([txt("File Upload: Users can click the 'Choose File' button to open a standard file browser dialog and select an image file from their device. The module accepts common image formats including JPEG, PNG, and BMP.", { size: 24 })]),
    justified([txt("Drag-and-Drop: The upload zone supports drag-and-drop functionality. Users can drag an image file from their file manager and drop it directly onto the designated area. Visual feedback is provided through hover effects and border changes during the drag operation.", { size: 24 })]),
    justified([txt("Camera Capture: The 'Open Camera' button activates the device camera using the getUserMedia Web API. A live video feed is displayed in the interface. Users can capture a snapshot by clicking the 'Capture' button, which draws the current video frame onto an HTML5 canvas element. The captured image is then converted to a Blob and submitted via FormData to the server.", { size: 24 })]),
    justified([txt("After image selection through any method, a preview is displayed to the user. Upon clicking 'Submit', the image is sent via a POST request to the /submit endpoint using the Fetch API with FormData encoding.", { size: 24 })]),

    heading2("4.2 CNN Prediction Module"),
    justified([txt("The CNN Prediction Module is the core intelligence of the system, responsible for classifying leaf images into one of 39 disease classes.", { size: 24 })]),
    justified([txt("Model Loading: The pre-trained model (plant_disease_model_1_latest.pt) is loaded at application startup using torch.load() with weights_only=True for security. The model is set to evaluation mode using model.eval(), which disables dropout during inference.", { size: 24 })]),
    justified([txt("Preprocessing Pipeline: The input image is opened using PIL, resized to 224x224 pixels, and converted to a PyTorch tensor using torchvision.transforms.functional.to_tensor(). The tensor is reshaped to batch format (-1, 3, 224, 224).", { size: 24 })]),
    justified([txt("Inference: The model performs a forward pass producing a 39-dimensional output vector. F.softmax() converts raw logits into probabilities. The top-3 predictions are extracted using torch.topk(). Confidence is expressed as a percentage.", { size: 24 })]),
    justified([txt("Severity Classification: A predefined SEVERITY_MAP assigns severity levels (Healthy, Mild, Moderate, Critical) to each disease class index. Healthy plant indices (3, 5, 7, 11, 15, 18, 20, 23, 24, 25, 28, 38) are classified as 'Healthy'.", { size: 24 })]),

    heading2("4.3 Results Display Module"),
    justified([txt("The Results Display Module (submit.html) presents diagnosis results including: disease identification with color-coded health status badge, animated confidence meter with percentage display, severity badge (Healthy/Mild/Moderate/Critical), alternative predictions when confidence is below 70%, disease description and prevention steps with translation support, supplement recommendation with product image and purchase link, and a 'Listen' button for text-to-speech output.", { size: 24 })]),

    heading2("4.4 Supplement Marketplace Module"),
    justified([txt("The Supplement Marketplace Module (market.html) provides a comprehensive listing of all 39 recommended agricultural products in a responsive grid layout. Client-side filtering allows browsing by category (Fertilizer for healthy plants, Supplement for diseased plants). Each product card displays the category badge, image, associated disease/plant name, product name, and a 'Buy Now' button linking to the external purchase page.", { size: 24 })]),

    heading2("4.5 Voice and Translation Module"),
    justified([txt("The Voice and Translation Module (voice.js) provides multilingual and accessibility features.", { size: 24 })]),
    justified([txt("Speech Recognition: The Web Speech API captures voice input for navigation commands (home, market, alerts) and disease/supplement search queries. Results are fetched from /api/search-disease and displayed in a modal overlay.", { size: 24 })]),
    justified([txt("Text-to-Speech: The SpeechSynthesis interface reads results aloud in the selected language with language-specific voice selection.", { size: 24 })]),
    justified([txt("Translation: Content marked with data-translatable is translated via /api/translate endpoint using deep-translator (Google Translator). Supports 10 languages: English, Hindi, Tamil, Telugu, Bengali, Marathi, Gujarati, Kannada, Malayalam, and Punjabi. Translations are cached.", { size: 24 })]),

    heading2("4.6 Community Alerts Module"),
    justified([txt("The Community Alerts Module enables farmers to report and monitor disease outbreaks regionally.", { size: 24 })]),
    justified([txt("Alert Reporting: After diagnosis, users can report findings to the community. The system captures geolocation via browser Geolocation API, reverse-geocodes via OpenStreetMap Nominatim, and stores alerts with disease details, severity, confidence, and location.", { size: 24 })]),
    justified([txt("Alert Visualization: The alerts page displays reported sightings on an interactive map. Color coding indicates severity: red (critical), orange (moderate), yellow (mild), green (healthy).", { size: 24 })]),
    justified([txt("Subscription System: Users subscribe to regional alerts by email. The system supports both Supabase (cloud) and SQLite (local fallback) backends.", { size: 24 })]),
  ];
}

// ── Chapter 5: Implementation ──────────────────────────────────────────────
function chapter5() {
  return [
    pageBreak(), heading1("Chapter 5: Implementation"),
    heading2("5.1 Technologies Used"),
    makeTable(["Technology", "Category", "Purpose"],
      [["Python 3.8+","Language","Backend development, model inference"],["Flask >= 2.3","Web Framework","HTTP routing, request handling"],["PyTorch >= 2.0","Deep Learning","CNN model architecture and inference"],["TorchVision","Image Processing","Image transformations (to_tensor)"],["Pillow >= 10.0","Image Library","Image loading and resizing"],["Pandas >= 2.0","Data Processing","CSV file reading"],["Deep Translator","Translation","Google Translator API"],["Supabase / SQLite","Database","Alerts and subscriptions"],["Gunicorn","WSGI Server","Production deployment"],["HTML5 / CSS3","Frontend","Page structure and styling"],["Bootstrap 5","CSS Framework","Responsive layout"],["JavaScript ES6","Frontend Logic","DOM, API calls, media access"],["Web Speech API","Browser API","Speech recognition and TTS"],["Jinja2","Template Engine","Server-side HTML rendering"],["Font Awesome 6.4","Icons","UI visual elements"]],
      [2200, 2200, 4626]),
    centered([txt("Table 5.1: Technologies Used", { size: 20, italics: true })], { before: 60, after: 200 }),

    heading2("5.2 Sample Code"),
    heading3("5.2.1 CNN Model Definition (CNN.py)"),
    justified([txt("The following code shows the core CNN model architecture:", { size: 24 })]),
    ...["class CNN(nn.Module):","    def __init__(self, K):","        super(CNN, self).__init__()","        self.conv_layers = nn.Sequential(","            nn.Conv2d(3, 32, kernel_size=3, padding=1),","            nn.ReLU(),","            nn.BatchNorm2d(32),","            nn.Conv2d(32, 32, kernel_size=3, padding=1),","            nn.ReLU(),","            nn.BatchNorm2d(32),","            nn.MaxPool2d(2),","            # ... Blocks 2-4 follow same pattern","        )","        self.dense_layers = nn.Sequential(","            nn.Dropout(0.4),","            nn.Linear(50176, 1024),","            nn.ReLU(),","            nn.Dropout(0.4),","            nn.Linear(1024, K),","        )"
    ].map(l => leftPara([txt(l, { size: 20 })], { after: 0, line: 240, indent: { left: 720 } })),
    emptyLine(),

    heading3("5.2.2 Prediction Pipeline (app.py)"),
    ...["def prediction(image_path):","    image = Image.open(image_path)","    image = image.resize((224, 224))","    input_data = TF.to_tensor(image)","    input_data = input_data.view((-1, 3, 224, 224))","    output = model(input_data)","    output = output.detach().numpy()","    index = output.argmax()","    return index"
    ].map(l => leftPara([txt(l, { size: 20 })], { after: 0, line: 240, indent: { left: 720 } })),
    emptyLine(),

    heading3("5.2.3 Flask Route - Submit (app.py)"),
    ...["@app.route('/submit', methods=['POST'])","def submit():","    image = request.files['image']","    filename = image.filename","    file_path = os.path.join('static/uploads', filename)","    image.save(file_path)","    pred = prediction(file_path)","    title = disease_info['disease_name'][pred]","    description = disease_info['description'][pred]","    return render_template('submit.html', ...)"
    ].map(l => leftPara([txt(l, { size: 20 })], { after: 0, line: 240, indent: { left: 720 } })),
    emptyLine(),

    heading2("5.3 Screenshots of Web Pages"),
    justified([txt("The following screenshots demonstrate the key pages of the application:", { size: 24 })]),
    ...["[Insert Figure 5.1: Home/Upload Page - main interface with drag-and-drop zone, camera button, voice button]",
      "[Insert Figure 5.2: Upload Page with Camera Active - live camera feed with capture button]",
      "[Insert Figure 5.3: Results Page - disease diagnosis with confidence meter, severity, description, supplement]",
      "[Insert Figure 5.4: Supplement Marketplace - product grid with filtering and buy links]",
      "[Insert Figure 5.5: Community Alerts Page - interactive map, alert cards, subscription form]"
    ].map(s => { return [justified([txt(s, { size: 24, italics: true, bold: true })]), emptyLine()]; }).flat(),

    heading2("5.4 Database Tables"),
    heading3("5.4.1 Disease Classes"),
    justified([txt("The system classifies images into 39 classes:", { size: 24 })]),
    makeTable(["Index", "Class Name", "Species"],
      [["0","Apple___Apple_scab","Apple"],["1","Apple___Black_rot","Apple"],["2","Apple___Cedar_apple_rust","Apple"],["3","Apple___healthy","Apple"],["4","Background_without_leaves","Background"],["5","Blueberry___healthy","Blueberry"],["6","Cherry___Powdery_mildew","Cherry"],["7","Cherry___healthy","Cherry"],["8","Corn___Cercospora_leaf_spot","Corn"],["9","Corn___Common_rust","Corn"],["10","Corn___Northern_Leaf_Blight","Corn"],["11","Corn___healthy","Corn"],["12","Grape___Black_rot","Grape"],["13","Grape___Esca_Black_Measles","Grape"],["14","Grape___Leaf_blight","Grape"],["15","Grape___healthy","Grape"],["16","Orange___Huanglongbing","Orange"],["17","Peach___Bacterial_spot","Peach"],["18","Peach___healthy","Peach"],["19","Pepper___Bacterial_spot","Pepper"],["20","Pepper___healthy","Pepper"],["21","Potato___Early_blight","Potato"],["22","Potato___Late_blight","Potato"],["23","Potato___healthy","Potato"],["24","Raspberry___healthy","Raspberry"],["25","Soybean___healthy","Soybean"],["26","Squash___Powdery_mildew","Squash"],["27","Strawberry___Leaf_scorch","Strawberry"],["28","Strawberry___healthy","Strawberry"],["29","Tomato___Bacterial_spot","Tomato"],["30","Tomato___Early_blight","Tomato"],["31","Tomato___Late_blight","Tomato"],["32","Tomato___Leaf_Mold","Tomato"],["33","Tomato___Septoria_leaf_spot","Tomato"],["34","Tomato___Spider_mites","Tomato"],["35","Tomato___Target_Spot","Tomato"],["36","Tomato___Yellow_Leaf_Curl_Virus","Tomato"],["37","Tomato___mosaic_virus","Tomato"],["38","Tomato___healthy","Tomato"]],
      [1000, 5026, 3000]),
    centered([txt("Table 5.2: Disease Classes and Indices", { size: 20, italics: true })], { before: 60, after: 200 }),

    heading3("5.4.2 Alerts Database Schema"),
    justified([txt("Table: disease_alerts", { size: 24, bold: true })]),
    makeTable(["Column", "Type", "Description"],
      [["id","INTEGER PRIMARY KEY","Auto-increment ID"],["disease_name","TEXT","Detected disease name"],["disease_index","INTEGER","Model index (0-38)"],["severity","TEXT","Severity level"],["confidence","REAL","Confidence (0.0-1.0)"],["latitude","REAL","GPS latitude"],["longitude","REAL","GPS longitude"],["region_name","TEXT","Reverse-geocoded region"],["image_url","TEXT","Uploaded image URL"],["reported_by","TEXT DEFAULT 'anonymous'","Reporter ID"],["created_at","TIMESTAMP","Creation timestamp"]],
      [2500, 3263, 3263]),
    centered([txt("Table 5.3: Database Schema - disease_alerts", { size: 20, italics: true })], { before: 60, after: 200 }),
    emptyLine(),
    justified([txt("Table: alert_subscriptions", { size: 24, bold: true })]),
    makeTable(["Column", "Type", "Description"],
      [["id","INTEGER PRIMARY KEY","Auto-increment ID"],["email","TEXT","Subscriber email"],["region_name","TEXT","Alert region"]],
      [2500, 3263, 3263]),
    centered([txt("Table 5.4: Database Schema - alert_subscriptions", { size: 20, italics: true })], { before: 60, after: 200 }),
  ];
}

// ── Chapter 6: Testing ─────────────────────────────────────────────────────
function chapter6() {
  return [
    pageBreak(), heading1("Chapter 6: Testing"),
    heading2("6.1 Testing Strategies"),
    heading3("6.1.1 Unit Testing"),
    justified([txt("Individual components were tested in isolation. The CNN model was tested with known input tensors to verify output dimensions (39 classes). Image preprocessing functions were tested for correct resizing and tensor conversion. CSV data loading was verified for all 39 entries.", { size: 24 })]),
    heading3("6.1.2 Integration Testing"),
    justified([txt("Integration tests verified component interactions. The end-to-end prediction pipeline was tested from image upload through preprocessing, model inference, and result retrieval. Flask route integration was tested for proper request handling and template rendering. Database operations were tested for both Supabase and SQLite backends.", { size: 24 })]),
    heading3("6.1.3 System Testing"),
    justified([txt("The complete system was tested as a whole. Multiple images from each disease class were uploaded to verify classification accuracy. Voice recognition and translation features were tested across different languages. The community alerts workflow was tested end-to-end.", { size: 24 })]),
    heading3("6.1.4 User Acceptance Testing"),
    justified([txt("The application was tested by potential end-users. Users uploaded images, interpreted results, and navigated between pages. Feedback was collected on interface clarity, response time, and overall experience. Multilingual features were tested by native speakers.", { size: 24 })]),

    heading2("6.2 Test Cases"),
    makeTable(["Test ID", "Description", "Input", "Expected Output", "Status"],
      [["TC01","Upload valid leaf image","JPEG tomato leaf","Disease prediction with confidence","Pass"],["TC02","Upload non-image file","PDF document","Error message","Pass"],["TC03","Camera capture and submit","Live camera frame","Disease prediction displayed","Pass"],["TC04","Healthy plant detection","Healthy apple leaf","Healthy status, green badge","Pass"],["TC05","Disease detection accuracy","Tomato early blight","Correct disease name and info","Pass"],["TC06","Confidence scoring","Clear disease image","High confidence (>80%)","Pass"],["TC07","Low confidence alternatives","Ambiguous leaf","Top-3 predictions shown","Pass"],["TC08","Language translation","Select Hindi","Content translated to Hindi","Pass"],["TC09","Voice command navigation","Say 'market'","Navigate to marketplace","Pass"],["TC10","Text-to-speech output","Click Listen button","Results read aloud","Pass"],["TC11","Marketplace filtering","Click Fertilizer tab","Only fertilizers shown","Pass"],["TC12","Report community alert","Click Report button","Alert saved with location","Pass"],["TC13","Subscribe to alerts","Enter email + region","Subscription confirmed","Pass"],["TC14","Drag-and-drop upload","Drag image to zone","Image preview displayed","Pass"],["TC15","Responsive design","Mobile viewport","UI renders correctly","Pass"],["TC16","Model accuracy","16 test images",">=90% correct","Pass"]],
      [900, 2200, 1800, 2400, 726]),
    centered([txt("Table 6.1: Test Cases", { size: 20, italics: true })], { before: 60, after: 200 }),

    heading2("6.3 Model Performance"),
    justified([txt("The CNN model was evaluated on the Plant Village dataset:", { size: 24 })]),
    ...["Training Set: 36,584 images - Accuracy: ~97%","Validation Set: 15,679 images - Accuracy: ~99%","Test Set: 9,223 images - Accuracy: ~99%"
    ].map(r => justified([txt(r, { size: 24 })], { indent: { left: 360 } })),
    emptyLine(),
    justified([txt("The model was trained for 5 epochs using Adam optimizer with CrossEntropyLoss and batch size 64. The high validation and test accuracy with minimal gap from training accuracy indicates good generalization, aided by Dropout(0.4) regularization and Batch Normalization.", { size: 24 })]),
  ];
}

// ── Chapter 7: Conclusion ──────────────────────────────────────────────────
function chapter7() {
  return [
    pageBreak(), heading1("Chapter 7: Conclusion"),
    heading2("7.1 Summary"),
    justified([txt("This project successfully designed and implemented an Advanced Deep Neural Network Framework for Early Plant Disease Identification and Precision Farming. The system employs a custom four-block CNN architecture trained on the Plant Village dataset containing 61,486 images across 39 disease classes and 14 plant species. The model achieves approximately 97% training accuracy and 99% validation/test accuracy.", { size: 24 })]),
    justified([txt("The framework is deployed as a comprehensive Flask web application providing multiple input methods, real-time diagnosis with confidence scoring and severity classification, actionable supplement recommendations, multilingual support for 10 Indian languages, text-to-speech accessibility, and a community-based alert system with geolocation mapping.", { size: 24 })]),

    heading2("7.2 Key Achievements"),
    ...["High Accuracy: 97-99% accuracy across training, validation, and test sets.","Comprehensive Coverage: 39 disease classes spanning 14 plant species.","Multi-Modal Input: File upload, drag-and-drop, camera capture, and voice commands.","Multilingual Accessibility: 10 Indian languages for diverse farming communities.","Actionable Intelligence: Severity assessment, confidence scoring, and supplement recommendations.","Community Features: Geolocation-based disease alert system for regional monitoring.","Responsive Design: Works seamlessly on desktop and mobile devices.","Cost-Effective: CPU-only infrastructure with no GPU requirement."
    ].map((a,i) => justified([txt(`${i+1}. ${a}`, { size: 24 })], { indent: { left: 360 } })),

    heading2("7.3 Future Enhancements"),
    ...["Extended Species Coverage: Additional crop species and diseases beyond current 14 species.","Native Mobile Application: Android/iOS apps with offline capability using PyTorch Mobile.","Real-Time Monitoring: IoT sensors and drone imagery for continuous field monitoring.","Transfer Learning: Fine-tuning pre-trained models (ResNet, EfficientNet) for improved accuracy.","Treatment Tracking: Logging treatments and tracking recovery progress.","Weather Integration: Disease risk prediction based on temperature, humidity, rainfall.","Expert Consultation: Direct connection with agricultural experts for complex cases.","Notification System: Push notifications for disease outbreaks in subscribed regions."
    ].map((e,i) => justified([txt(`${i+1}. ${e}`, { size: 24 })], { indent: { left: 360 } })),
  ];
}

// ── Bibliography ───────────────────────────────────────────────────────────
function bibliography() {
  return [
    pageBreak(), heading1("Bibliography"), emptyLine(),
    ...[
      "[1] Hughes, D.P. and Salathe, M. (2015). An open access repository of images on plant health to enable the development of mobile disease diagnostics. arXiv preprint arXiv:1511.08060.",
      "[2] Mohanty, S.P., Hughes, D.P. and Salathe, M. (2016). Using Deep Learning for Image-Based Plant Disease Detection. Frontiers in Plant Science, 7, p.1419.",
      "[3] LeCun, Y., Bengio, Y. and Hinton, G. (2015). Deep learning. Nature, 521(7553), pp.436-444.",
      "[4] Krizhevsky, A., Sutskever, I. and Hinton, G.E. (2012). ImageNet Classification with Deep Convolutional Neural Networks. NIPS, 25.",
      "[5] PyTorch Documentation. https://pytorch.org/docs/stable/",
      "[6] Flask Documentation. https://flask.palletsprojects.com/",
      "[7] Bootstrap 5 Documentation. https://getbootstrap.com/docs/5.0/",
      "[8] Web Speech API - MDN Web Docs. https://developer.mozilla.org/en-US/docs/Web/API/Web_Speech_API",
      "[9] FAO. Plant Pests and Diseases. https://www.fao.org/",
      "[10] Ioffe, S. and Szegedy, C. (2015). Batch Normalization: Accelerating Deep Network Training. ICML.",
      "[11] Srivastava, N. et al. (2014). Dropout: A Simple Way to Prevent Neural Networks from Overfitting. JMLR, 15(1), pp.1929-1958.",
      "[12] Ferentinos, K.P. (2018). Deep learning models for plant disease detection and diagnosis. Computers and Electronics in Agriculture, 145, pp.311-318.",
    ].map(ref => justified([txt(ref, { size: 22 })], { indent: { left: 720, hanging: 720 }, after: 160 })),
  ];
}

// ── Build Document ─────────────────────────────────────────────────────────
async function main() {
  const doc = new Document({
    styles: {
      default: { document: { run: { font: FONT, size: 24 } } },
      paragraphStyles: [
        { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { size: 32, bold: true, font: FONT },
          paragraph: { spacing: { before: 240, after: 240 }, alignment: AlignmentType.CENTER, outlineLevel: 0 } },
        { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { size: 28, bold: true, font: FONT },
          paragraph: { spacing: { before: 200, after: 160 }, outlineLevel: 1 } },
        { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { size: 26, bold: true, italics: true, font: FONT },
          paragraph: { spacing: { before: 160, after: 120 }, outlineLevel: 2 } },
      ]
    },
    sections: [
      // Section 1: Preliminary pages (NO page numbers)
      {
        properties: {
          page: {
            size: { width: PAGE_WIDTH, height: PAGE_HEIGHT },
            margin: { top: MARGIN, right: MARGIN, bottom: MARGIN, left: MARGIN },
          },
          titlePage: true,
        },
        children: [
          ...titlePage(), ...certificatePage(), ...acknowledgementPage(),
          ...abstractPage(), ...tocPage(), ...listOfTablesPage(), ...listOfFiguresPage(),
        ],
      },
      // Section 2: Main content (WITH page numbers)
      {
        properties: {
          type: SectionType.NEXT_PAGE,
          page: {
            size: { width: PAGE_WIDTH, height: PAGE_HEIGHT },
            margin: { top: MARGIN, right: MARGIN, bottom: MARGIN, left: MARGIN },
          },
        },
        headers: {
          default: new Header({
            children: [new Paragraph({
              alignment: AlignmentType.CENTER,
              border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: "999999", space: 1 } },
              children: [new TextRun({ text: "Advanced Deep Neural Network Framework for Plant Disease Identification", font: FONT, size: 18, italics: true })]
            })]
          })
        },
        footers: {
          default: new Footer({
            children: [new Paragraph({
              alignment: AlignmentType.CENTER,
              children: [new TextRun({ text: "Page ", font: FONT, size: 20 }), new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 20 })]
            })]
          })
        },
        children: [
          ...chapter1(), ...chapter2(), ...chapter3(), ...chapter4(),
          ...chapter5(), ...chapter6(), ...chapter7(), ...bibliography(),
        ],
      }
    ],
  });

  const buffer = await Packer.toBuffer(doc);
  fs.writeFileSync("Plant_Disease_Project_Report.docx", buffer);
  console.log("Document generated: Plant_Disease_Project_Report.docx");
  console.log(`Size: ${(buffer.length / 1024).toFixed(1)} KB`);
}

main().catch(err => { console.error("Error:", err); process.exit(1); });
