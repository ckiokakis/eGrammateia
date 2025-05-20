import type { NextApiRequest, NextApiResponse } from 'next';
import fs from 'fs';
import path from 'path';

export default function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method === 'GET') {
    const dir = path.join(process.cwd(), 'documents');
    if (!fs.existsSync(dir)) {
      return res.status(200).json({ files: [] });
    }
    const files = fs.readdirSync(dir);
    return res.status(200).json({ files });
  } else {
    res.setHeader('Allow', 'GET');
    res.status(405).end('Method Not Allowed');
  }
}

