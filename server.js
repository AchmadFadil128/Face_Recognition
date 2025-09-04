const express = require('express');
const bodyParser = require('body-parser');
const fs = require('fs');
const path = require('path');
const cors = require('cors');

const DATA_FILE = path.join(__dirname, 'students.json');
const PORT = process.env.PORT || 3000;

const app = express();
app.use(cors());
app.use(bodyParser.json());
app.use(express.static(path.join(__dirname, 'public'))); // serve index.html from /public

// SSE clients
let clients = [];

// load students from file or seed
function loadStudents() {
  try {
    if (fs.existsSync(DATA_FILE)) {
      const content = fs.readFileSync(DATA_FILE, 'utf8');
      return JSON.parse(content);
    }
  } catch (err) {
    console.error("Failed load students:", err);
  }
  // default seed
  return [
    { studentNumber: "SW001", name: "Andi", class: "XIPA1", status: "ABSENT" },
    { studentNumber: "SW002", name: "Budi", class: "XIPA1", status: "ABSENT" },
    { studentNumber: "SW003", name: "Citra", class: "XIPA2", status: "ABSENT" }
  ];
}

function saveStudents(students) {
  fs.writeFileSync(DATA_FILE, JSON.stringify(students, null, 2));
}

let students = loadStudents();
saveStudents(students);

// SSE endpoint
app.get('/events', (req, res) => {
  res.set({
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache',
    'Connection': 'keep-alive'
  });
  res.flushHeaders();

  const clientId = Date.now();
  const newClient = {
    id: clientId,
    res
  };
  clients.push(newClient);
  console.log(`Client connected: ${clientId}. Total clients: ${clients.length}`);

  // send initial data
  res.write(`event: students\n`);
  res.write(`data: ${JSON.stringify(students)}\n\n`);

  req.on('close', () => {
    clients = clients.filter(c => c.id !== clientId);
    console.log(`Client disconnected: ${clientId}. Remaining: ${clients.length}`);
  });
});

function sendEvent(eventName, payload) {
  const data = `event: ${eventName}\ndata: ${JSON.stringify(payload)}\n\n`;
  clients.forEach(c => c.res.write(data));
}

// API: get list
app.get('/api/students', (req, res) => {
  res.json(students);
});

// API: add student
app.post('/api/students', (req, res) => {
  const { studentNumber, name, class: klass } = req.body;
  if (!studentNumber || !name) {
    return res.status(400).json({ error: "studentNumber and name are required" });
  }
  if (students.find(s => s.studentNumber === studentNumber)) {
    return res.status(409).json({ error: "studentNumber already exists" });
  }
  const newStudent = { studentNumber, name, class: klass || '', status: "ABSENT" };
  students.push(newStudent);
  saveStudents(students);
  sendEvent('students', students);
  res.json(newStudent);
});

// API: delete student
app.delete('/api/students/:studentNumber', (req, res) => {
  const id = req.params.studentNumber;
  const before = students.length;
  students = students.filter(s => s.studentNumber !== id);
  if (students.length === before) {
    return res.status(404).json({ error: "not found" });
  }
  saveStudents(students);
  sendEvent('students', students);
  res.json({ success: true });
});

// API: mark attendance (the endpoint your Python script will hit)
app.post('/api/attendance/mark', (req, res) => {
  const { studentNumber, status } = req.body;
  if (!studentNumber || !status) {
    return res.status(400).json({ error: "studentNumber and status are required" });
  }
  const normalizedStatus = String(status).toUpperCase();
  if (!["PRESENT","ABSENT"].includes(normalizedStatus)) {
    return res.status(400).json({ error: "status must be PRESENT or ABSENT" });
  }

  let student = students.find(s => s.studentNumber === studentNumber);
  if (!student) {
    // Option A: auto-create placeholder
    student = { studentNumber, name: "Unknown", class: "", status: normalizedStatus };
    students.push(student);
  } else {
    student.status = normalizedStatus;
  }
  saveStudents(students);
  // send a dedicated attendance event and full student list too
  sendEvent('attendance', { studentNumber, status: normalizedStatus });
  sendEvent('students', students);

  res.json({ ok: true, student });
});

app.listen(PORT, () => {
  console.log(`Attendance demo server listening on http://localhost:${PORT}`);
  console.log(`Open http://localhost:${PORT} in your browser (index.html)`);
});
