import type { NextApiRequest, NextApiResponse } from 'next';
import fs from 'fs';
import path from 'path';

export default function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method === 'DELETE') {
    const { filename } = req.body;
    const filePath = path.join(process.cwd(), 'documents', filename);
    if (fs.existsSync(filePath)) {
      fs.unlinkSync(filePath);
      return res.status(200).json({ success: true });
    }
    return res.status(404).json({ error: 'File not found' });
  } else {
    res.setHeader('Allow', 'DELETE');
    res.status(405).end('Method Not Allowed');
  }
}