import { createBrowserRouter } from 'react-router-dom';
import Layout from '../shared/components/Layout';
import Dashboard from '../routes/dashboard';
import CoinChecker from '../routes/coin_checker';
import ComHub from '../routes/com_hub';
import DeepfakeDetector from '../routes/deepfake_detector';
import News from '../routes/news';
import ScamReport from '../routes/scam_report';
import WalletAnalyzer from '../routes/wallet_analyzer';

export const router = createBrowserRouter([
  {
    path: '/',
    element: <Layout />,
    children: [
      {
        index: true,
        element: <Dashboard />,
      },
      {
        path: 'dashboard',
        element: <Dashboard />,
      },
      {
        path: 'coin-checker',
        element: <CoinChecker />,
      },
      {
        path: 'com-hub',
        element: <ComHub />,
      },
      {
        path: 'deepfake-detector',
        element: <DeepfakeDetector />,
      },
      {
        path: 'news',
        element: <News />,
      },
      {
        path: 'scam-report',
        element: <ScamReport />,
      },
      {
        path: 'wallet-analyzer',
        element: <WalletAnalyzer />,
      },
    ],
  },
]);

