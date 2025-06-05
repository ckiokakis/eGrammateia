// import type { NextApiRequest, NextApiResponse } from 'next';
// import formidable from 'formidable';
// import fs from 'fs';
// import path from 'path';

// export const config = { api: { bodyParser: false } };

// const uploadDir = path.join(process.cwd(), 'documents');
// if (!fs.existsSync(uploadDir)) fs.mkdirSync(uploadDir);

// export default function handler(req: NextApiRequest, res: NextApiResponse) {
//   if (req.method === 'POST') {
//     const form = new formidable.IncomingForm({ uploadDir, keepExtensions: true });
//     form.parse(req, (err, fields, files) => {
//       if (err) return res.status(500).json({ error: err.message });
//       return res.status(200).json({ success: true });
//     });
//   } else {
//     res.setHeader('Allow', 'POST');
//     res.status(405).end('Method Not Allowed');
//   }
// }
